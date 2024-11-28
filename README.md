# Fair Cognitive Diagnosis Without Sensitive Attributes (FairWISA)

The code for the paper **"Fair Cognitive Diagnosis Without Sensitive Attributes" (FairWISA)**.

The full version of the code is coming soon.

## Data
PISA data is available at: [PISA Data](https://www.oecd.org/pisa/data/)

SLP data is available at: [SLP Data](https://aic-fe.bnu.edu.cn/en/data/index.html)

## Environment Requirements
- **EduCDM**
- **torch**: 1.13.1  
- **pandas**: 1.0.1  
- **scipy**: 1.4.1  
- **numpy**: 1.21.6  
- **tensorboardX**: 2.6  
- **scikit-learn**: 0.22.1  
- **tqdm**: 4.42.1  

## How It Runs
1. Train a Cognitive Diagnosis (CD) model (e.g., IRT) to get the model parameters file: `model.params`.
2. Execute the following command to generate the grouping matrix file: `groups.params`:
   ```bash
   python gengroups.py
3. Execute the following command to obtain the fair CD model:
   ```bash
   python test
