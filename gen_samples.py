import torch

EPS = 1e-7
#Simple two variable case (X_1, X_2)
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


def KL_div(d1, d2): 
    #two distributions
    assert d1.shape == d2.shape, "Two distributions must have the same dimension"

    d1 += EPS 
    d2 += EPS
    tmp = d1 * torch.log(d1/d2) #base-e logarithm
    return torch.sum(tmp)

def jensen_shannon(d1, d2): 
    #symmetric in d1, d2
    m = 0.5 * (d1+d2)
    return 0.5*KL_div(d1, m) + 0.5*KL_div(d2, m)

##Gibbs Sampling Markov Chain Monte Carlo
def gibbs_samp_mcmc(file_path: str, num_samples, conds, true_joint):
    #expected: conds is shape X_i | X_.. -> so n * l^n
    #i.e. if 3 variables, then P[1, a, b,c] = P(X_1 = a | X_2 = b, X_3 = c)
    #P[2, a, b,c] = P(X_2 = b | X_1 = a, X_3 = c) 
    
    if true_joint == "none": 
        #assuming u passed in the conditionals
        n, l = conds.shape[0], conds.shape[1];
    else: 
        #recover conds from true_joint for testing purposes
        l, n = true_joint.shape[0], true_joint.dim()
        for i in range(n): 

            #getting P(X_i | X_1..X_i-1, X_i+1...X_n), fixing X_i = val
            for val in range(l):
                joint_idx_i = [slice(None) if j != i else val for j in range(n)]
                conds_idx_i = [i if j == 0 else val if j == i + 1 else slice(None) for j in range(n+1)]
                margs = true_joint.sum(dim=i, keepdim=False)

                conds[tuple(conds_idx_i)] = true_joint[tuple(joint_idx_i)] / margs
        #print("computed conditional distr")
        #print(conds)

    #tracking X_1,X_2,...,X_n
    cur_state = torch.zeros(n, dtype=torch.long)
    freqs = torch.zeros(*([l] * n), dtype=torch.long)

    its = 500
    for samp in range(num_samples):
        for t in range(its): 
            #sample each X_i from the previous

            for i in range(n): 
                index = [i]
                for j in range(n): 
                    if j == i: index.append(slice(None))
                    else: index.append(cur_state[j].item())
                i_distr = conds[tuple(index)]
                i_distr = i_distr/i_distr.sum() #normalize for safety (shld not be needed)
                i_val = torch.multinomial(i_distr, 1)
                cur_state[i] = i_val

        #print("Sample ", samp, ": ", cur_state)
        #write to freqs
        res = [i for i in cur_state]
        freqs[tuple(res)] += 1

    freqs = freqs/num_samples
    #print("Overall frequencies ")
    #print(freqs)

    div = jensen_shannon(freqs, true_joint)
    return freqs, div

#write_samples2("datasets/samp/samp.train.data", 20, true_joint)
#write_samples2("datasets/samp/samp.valid.data", 10, true_joint)
#write_samples2("datasets/samp/samp.test.data", 10, true_joint)

#### Gibbs MCMC testing with joints (2 variable, 3-category case)
x12_joint = torch.rand(3, 3)
x12_joint = x12_joint / x12_joint.sum()
print("true joint is: ")
print(x12_joint)
gibbs_samp_mcmc("temp", 100, torch.zeros(2,3,3), x12_joint)

#testing divergence as num of iterations varies 
for its in range(50, 400, 50): 
    _, div = gibbs_samp_mcmc("temp", its, torch.zeros(2,3,3,), x12_joint)
    print(its, " iterations, Jensen-Shannon div: ", div)





