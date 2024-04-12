FILE = '../FinPROTAX/FinPROTAX-main/FinPROTAX/modelCOIfull/refs.aln' # path to reference data (change as needed)

def k_mers(seq, k):
    k_mers_tokens = []
    for i in range(len(seq)):
        if i + k > len(seq):
            break
        k_mers_tokens.append(seq[i:i+k])

    return k_mers_tokens

# sample usage
if __name__ == '__main__':
    k = 10
    with open(FILE, "r") as file:
        lines = file.readlines()

    dna_sequences = []

    for line in lines:
        if line.startswith('-'):
            s = line.replace('-', '')
            dna_sequences.append(s.strip())
    
    tokens = []

    for seq in dna_sequences:
        tokens.extend(k_mers(seq, k))
    
    # print(tokens)