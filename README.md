# Instructions and File Structure

## Methodology:
This project focuses on generating GenZ style emails and classifying email hierarchy using various machine learning techniques. <br>
***Hypothesis***: GenZ style emails have distinct linguistic patterns and more informal tone compared to traditional emails. So models trained on traditional email datasets may not perform well on GenZ style emails. 

***For the detailed methodology, approach, and experiments, please refer to the [Final_Report.pdf](./Final_Report.pdf) file in the root directory.***

## Preprocessing the Data:
Use the `preprocess.ipynb` to preprocess the data. This notebook will :
- Load the data from the `datasets/emails.csv` file.
- Preprocess the data and structuring the email data into a format suitable for training.
- Save the preprocessed data into a new file `datasets/emails_cleaned.csv`.
- Since the data is large, we only use a small sample for further tasks and save it in `datasets/sample_emails.csv`.
- Now the Hierarchial Order is found using SNA and saved in the same file.


## GenZ Data Generation : `Data Generation\` folder
- `dataset.ipynb` is a basic approach
- `genz_data_gen.ipynb` and `genz_twitter.ipynb` are LLM based approaches
- `genz_transformer.ipynb` is a transformer based approach that uses the samples obtained from LLM based approaches to train a transformer model to generate GenZ data.
- The api keys for groq api should be added in `genz_twitter.ipynb` and `genz_transformer.ipynb` files to run the code.

### Getting API Keys 
```python
api_keys = [
    "your_groq_api_key_1",
    "your_groq_api_key_2",
    # Add more API keys as needed
]
```

#### ***For sake of reference here's the link to get Groq API keys: [Groq API Keys](https://console.groq.com/keys)***


## Models
- `rnn.ipynb` is a basic RNN model that uses the preprocessed data to train a model to classify the emails using the contents of the email.
- `lstm.py` is a LSTM model that uses the preprocessed data to train a model to classify the emails using the contents of the email.
- `BERT.ipynb` is a transformer based model that uses the preprocessed data to train a model to classify the emails using the contents of the email. Saved in `email_hierarchy_model` folder.



## One Drive Link 
- [One Drive Link](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/naga_revanth_students_iiit_ac_in/EuFXwjbdmx5PsQ_z5SYRy9EBb36xOl3FaqTvD2lhC_6zBg?e=tkaEvR) to download the datasets and models.