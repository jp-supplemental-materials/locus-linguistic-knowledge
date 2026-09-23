# The Locus of Linguistic Knowledge

Supplementary simulation code investigating phonetic neutralization in exemplar-based models of speech production.

## Models

- **Model 1:** Exemplar selection, entrenchment, noise, storage, and decay.
- **Model 2:** Adds a pre-nasal production bias.
- **Model 3:** Incorporates production-bias optimization.
- **Model 4:** Examines reduced oral-category frequency.
- **Model 5.1:** Examines high nasality.
- **Model 5.2:** Examines socially weighted encoding.
- **Model 6:** Examines production-bias strength modulation.

## Running the code

The scripts use Python 3. Install the required packages:

```bash
python3 -m pip install numpy pandas scipy matplotlib
```

Run an individual model, for example:

```bash
python3 model6_bias_strength_modulation.py
```

Simulation parameters are specified within each script. Results vary slightly across runs because the simulations include random sampling, but general patterns are consistent.
