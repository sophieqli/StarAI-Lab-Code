import torch


#The two variable case (X_1, X_2)
def write_samples2(file_path: str, num_samples, true_joint):
    # Sample num_samples points
    flat_probs = true_joint.flatten()
    samples_flat = torch.multinomial(flat_probs, num_samples, replacement=True)

    x_samples = (samples_flat // 2).tolist()
    y_samples = (samples_flat % 2).tolist()

    # Write to samp.train.data format
    with open(file_path, "w") as f:
        for idx, (x, y) in enumerate(zip(x_samples, y_samples), 1):
            line = f"{x},{y}\n"
            f.write(line)

# Known 2x2 joint distribution over X and Y
true_joint = torch.tensor([[0.1, 0.4],[0.2, 0.3]])  # shape (2, 2)

# The multi-variate case (X_1, X_2, ... X_n) 
# I learned about and implemented various sampling algos for fun!

def multi_sample_gibbs(file_path: str, num_samples, true_joint):


write_samples2("datasets/samp/samp.train.data", 20, true_joint)
write_samples2("datasets/samp/samp.valid.data", 10, true_joint)
write_samples2("datasets/samp/samp.test.data", 10, true_joint)


print("done writing to train, valid, and testdata! yipee!")

