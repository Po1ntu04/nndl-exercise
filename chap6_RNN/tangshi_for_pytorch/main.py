from __future__ import annotations

import argparse
import collections
import random
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

import rnn

START_TOKEN = "G"
END_TOKEN = "E"
BEGIN_WORDS = ["日", "红", "山", "夜", "湖", "海", "月"]

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "poems.txt"
ALT_DATA_FILE = BASE_DIR / "tangshi.txt"
FIG_DIR = BASE_DIR / "figures"

DEFAULT_BATCH_SIZE = 100
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
DEFAULT_EPOCHS = 30
LEARNING_RATE = 0.01
MAX_GEN_LEN = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def get_model_artifacts(model_type):
    model_type = model_type.lower()
    if model_type == "lstm":
        suffix = "lstm"
    elif model_type == "rnn":
        suffix = "rnn"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return {
        "checkpoint": BASE_DIR / f"poem_generator_{suffix}.pt",
        "loss_curve": FIG_DIR / f"training_loss_curve_{suffix}.png",
        "poem_panel": FIG_DIR / f"generated_poems_panel_{suffix}.png",
        "poem_text": BASE_DIR / f"generated_poems_{suffix}.txt",
    }


def configure_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_content(content):
    return content.replace(" ", "").replace("\u3000", "")


def is_valid_poem(content):
    invalid_marks = {"_", "(", "（", "[", "【", "<", "《", START_TOKEN, END_TOKEN}
    if any(mark in content for mark in invalid_marks):
        return False
    return 5 <= len(content) <= 80


def split_title_and_content(line):
    if ":" in line:
        return line.split(":", 1)
    if "：" in line:
        return line.split("：", 1)
    return None, None


def process_poems1(file_name):
    poems = []
    with open(file_name, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            _, content = split_title_and_content(line)
            if content is None:
                continue
            content = clean_content(content)
            if not is_valid_poem(content):
                continue
            poems.append(START_TOKEN + content + END_TOKEN)

    poems = sorted(poems, key=len)
    all_words = []
    for poem in poems:
        all_words.extend(poem)

    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[: len(words)] + (" ",)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def process_poems2(file_name):
    poems = []
    with open(file_name, "r", encoding="utf-8") as f:
        for raw_line in f:
            content = clean_content(raw_line.strip())
            if not content or not is_valid_poem(content):
                continue
            poems.append(START_TOKEN + content + END_TOKEN)

    poems = sorted(poems, key=len)
    all_words = []
    for poem in poems:
        all_words.extend(poem)

    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[: len(words)] + (" ",)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, pad_id):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            target = row[1:]
            target.append(row[-1])
            y_data.append(target)
        max_len = max(len(row) for row in x_data)
        x_data = [row + [pad_id] * (max_len - len(row)) for row in x_data]
        y_data = [row + [pad_id] * (max_len - len(row)) for row in y_data]
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def build_model(batch_size, vocab_size, model_type):
    word_embedding = rnn.word_embedding(vocab_length=vocab_size, embedding_dim=EMBEDDING_DIM)
    model = rnn.RNN_model(
        batch_sz=batch_size,
        vocab_len=vocab_size,
        word_embedding=word_embedding,
        embedding_dim=EMBEDDING_DIM,
        lstm_hidden_dim=HIDDEN_DIM,
        model_type=model_type,
    )
    return model.to(DEVICE)


def exponential_smooth(values, alpha=0.15):
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed


def plot_training_history(history, save_path, model_type):
    configure_plot_style()
    batch_losses = history["batch_losses"]
    epoch_losses = history["epoch_losses"]
    batch_steps = np.arange(1, len(batch_losses) + 1)
    epoch_steps = np.arange(1, len(epoch_losses) + 1)
    smooth_losses = exponential_smooth(batch_losses)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    axes[0].plot(batch_steps, batch_losses, color="#7aa6c2", linewidth=1.2, alpha=0.55, label="Batch loss")
    axes[0].plot(batch_steps, smooth_losses, color="#0b5d8a", linewidth=2.0, label="Smoothed loss")
    axes[0].set_title(f"{model_type.upper()} Training Loss per Batch")
    axes[0].set_xlabel("Batch step")
    axes[0].set_ylabel("NLL loss")
    axes[0].legend(frameon=True)

    axes[1].plot(epoch_steps, epoch_losses, color="#b63c3c", linewidth=2.0, marker="o", markersize=3.2)
    axes[1].set_title(f"{model_type.upper()} Mean Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean NLL loss")

    best_epoch = int(np.argmin(epoch_losses)) + 1
    best_loss = float(np.min(epoch_losses))
    fig.suptitle(
        f"{model_type.upper()} Poem Generator Training Curves | best epoch = {best_epoch}, best loss = {best_loss:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def format_poem(poem):
    content = poem.replace(START_TOKEN, "").replace(END_TOKEN, "")
    if "。" in content:
        lines = [line for line in content.split("。") if line]
        return "。\n".join(lines) + ("。" if lines else "")
    return "\n".join(textwrap.wrap(content, width=12))


def save_poem_panel(poems_by_word, save_path, model_type):
    configure_plot_style()
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), dpi=300)
    axes = axes.flatten()

    for ax, (begin_word, poem) in zip(axes, poems_by_word.items()):
        ax.set_axis_off()
        ax.set_title(f"{model_type.upper()} | begin word: {begin_word}", fontsize=12, pad=10)
        ax.text(
            0.02,
            0.95,
            format_poem(poem),
            va="top",
            ha="left",
            fontsize=12,
            linespacing=1.6,
            family="Microsoft YaHei",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f3e8", edgecolor="#7f6b45", linewidth=1.0),
        )

    if len(axes) > len(poems_by_word):
        axes[-1].set_axis_off()

    fig.suptitle(f"Generated Poems with Specified Begin Words ({model_type.upper()})", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_training(data_file=DATA_FILE, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, model_type="lstm"):
    artifacts = get_model_artifacts(model_type)
    poems_vector, word_to_int, _ = process_poems1(data_file)
    pad_id = word_to_int[" "]
    print(f"finish loading data from {data_file.name}")
    print(f"device: {DEVICE}")
    print(f"model: {model_type.upper()}")
    print(f"poems: {len(poems_vector)}, vocab size: {len(word_to_int) + 1}")

    model = build_model(batch_size=batch_size, vocab_size=len(word_to_int) + 1, model_type=model_type)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    loss_fun = torch.nn.NLLLoss(ignore_index=pad_id)
    history = {"batch_losses": [], "epoch_losses": []}

    for epoch in range(epochs):
        batches_inputs, batches_outputs = generate_batch(batch_size, poems_vector, pad_id)
        batch_order = np.random.permutation(len(batches_inputs))
        epoch_loss_sum = 0.0

        for batch_idx, order_idx in enumerate(batch_order):
            batch_x = np.array(batches_inputs[order_idx], dtype=np.int64)
            batch_y = np.array(batches_outputs[order_idx], dtype=np.int64)
            x_tensor = torch.from_numpy(batch_x).to(DEVICE)
            y_tensor = torch.from_numpy(batch_y).to(DEVICE)

            pred = model(x_tensor)
            batch_loss = loss_fun(pred, y_tensor.view(-1))

            if batch_idx % 100 == 0:
                _, pred_ids = torch.max(pred, dim=1)
                print("prediction", pred_ids[: min(24, pred_ids.numel())].detach().cpu().tolist())
                print("b_y       ", y_tensor.view(-1)[: min(24, y_tensor.numel())].detach().cpu().tolist())
                print("*" * 30)

            loss_value = float(batch_loss.item())
            history["batch_losses"].append(loss_value)
            epoch_loss_sum += loss_value

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_idx % 20 == 0:
                torch.save(model.state_dict(), artifacts["checkpoint"])
                print(
                    f"{model_type.upper()} | epoch {epoch:02d} | batch {batch_idx:03d} | "
                    f"loss = {loss_value:.4f} | checkpoint saved"
                )

        epoch_mean_loss = epoch_loss_sum / max(len(batches_inputs), 1)
        history["epoch_losses"].append(epoch_mean_loss)
        print(f"{model_type.upper()} [epoch {epoch + 1:02d}/{epochs:02d}] mean loss = {epoch_mean_loss:.4f}")

    torch.save(model.state_dict(), artifacts["checkpoint"])
    plot_training_history(history, artifacts["loss_curve"], model_type)
    return history


def to_word(predict, vocabs):
    sample = int(np.argmax(predict))
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def gen_poem(begin_word, data_file=DATA_FILE, model_type="lstm"):
    artifacts = get_model_artifacts(model_type)
    _, word_int_map, vocabularies = process_poems1(data_file)
    model = build_model(batch_size=64, vocab_size=len(word_int_map) + 1, model_type=model_type)
    model.load_state_dict(torch.load(artifacts["checkpoint"], map_location=DEVICE))
    model.eval()

    poem = begin_word
    word = begin_word
    with torch.no_grad():
        while word != END_TOKEN:
            input_ids = np.array([word_int_map.get(w, 0) for w in poem], dtype=np.int64)
            input_tensor = torch.from_numpy(input_ids).to(DEVICE)
            output = model(input_tensor, is_test=True)
            word = to_word(output.detach().cpu().numpy()[-1], vocabularies)
            poem += word
            if len(poem) > MAX_GEN_LEN:
                break
    return poem


def generate_all_poems(begin_words=BEGIN_WORDS, data_file=DATA_FILE, model_type="lstm"):
    artifacts = get_model_artifacts(model_type)
    poems_by_word = {}
    for begin_word in begin_words:
        poems_by_word[begin_word] = gen_poem(begin_word, data_file=data_file, model_type=model_type)

    save_poem_panel(poems_by_word, artifacts["poem_panel"], model_type)
    with open(artifacts["poem_text"], "w", encoding="utf-8") as f:
        for begin_word, poem in poems_by_word.items():
            formatted = format_poem(poem)
            print(f"[{begin_word}]\n{formatted}\n")
            f.write(f"[{begin_word}]\n{formatted}\n\n")
    return poems_by_word


def parse_args():
    parser = argparse.ArgumentParser(description="Tang poetry generation with RNN/LSTM")
    parser.add_argument("--mode", choices=["train", "generate", "all"], default="all")
    parser.add_argument("--model-type", choices=["lstm", "rnn"], default="lstm")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--use-alt-data", action="store_true", help="Use tangshi.txt instead of poems.txt")
    return parser.parse_args()


if __name__ == "__main__":
    set_seed(5)
    args = parse_args()
    data_file = ALT_DATA_FILE if args.use_alt_data else DATA_FILE
    artifacts = get_model_artifacts(args.model_type)

    if args.mode in {"train", "all"}:
        run_training(
            data_file=data_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_type=args.model_type,
        )

    if args.mode in {"generate", "all"}:
        if not artifacts["checkpoint"].exists():
            raise FileNotFoundError(f"Checkpoint not found: {artifacts['checkpoint']}")
        generate_all_poems(data_file=data_file, model_type=args.model_type)
