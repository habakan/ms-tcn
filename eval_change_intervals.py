import numpy as np
import argparse


def read_file(path):
    with open(path, "r") as f:
        content = f.read()
        f.close()
    return content


def get_change_indices(frame_wise_labels):
    last_label = frame_wise_labels[0]
    change_indices = []
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            change_indices.append(i)
            last_label = frame_wise_labels[i]
    return change_indices


def get_eval_indices(change_indices, neighbor_frames):
    eval_indices = []
    for change_index in change_indices:
        eval_indices += list(
            range(change_index - neighbor_frames, change_index + neighbor_frames)
        )
    eval_indices = [index for index in list(set(eval_indices)) if index > 0]
    return eval_indices


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="gtea")
    parser.add_argument("--neighbor_frames", type=int, default=10)
    parser.add_argument("--split", default="1")

    args = parser.parse_args()

    ground_truth_path = "./data/" + args.dataset + "/groundTruth/"
    recog_path = "./results/" + args.dataset + "/split_" + args.split + "/"
    file_list = "./data/" + args.dataset + "/splits/test.split" + args.split + ".bundle"

    list_of_videos = read_file(file_list).split("\n")[:-1]

    correct = 0
    total = 0

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split("\n")[0:-1]

        recog_file = recog_path + vid.split(".")[0]
        recog_content = read_file(recog_file).split("\n")[1].split()

        change_indices = get_change_indices(gt_content)
        eval_indices = get_eval_indices(change_indices, args.neighbor_frames)

        for i in eval_indices:
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

    print("Acc: %.4f" % (100 * float(correct) / total))


if __name__ == "__main__":
    main()
