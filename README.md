# Book Genre Prediction

## Project Overview
This project focuses on predicting the genre of a book based on its description using Natural Language Processing (NLP) techniques. The goal is to help readers find books they are more likely to enjoy by accurately classifying books into their respective genres. We fine-tuned multiple pre-trained language models (BERT, RoBERTa, and DistilBERT) to achieve this task.

## Team Members
•⁠  ⁠Anastasia Kurakova
•⁠  ⁠Sabnam Pandit

## Project Roadmap
1.⁠ ⁠*Data Collection*: Collected book data from ⁠ books.toscrape.com ⁠ and supplemented it with a Kaggle dataset.
2.⁠ ⁠*Data Cleaning*: Combined datasets, removed duplicates, and generalized some genres to improve prediction quality.
3.⁠ ⁠*Model Training*: Fine-tuned BERT, RoBERTa, and DistilBERT models using the Hugging Face Trainer API.
4.⁠ ⁠*Evaluation*: Evaluated model performance using accuracy metrics and tested on various book descriptions.
5.⁠ ⁠*Results*: Achieved an accuracy of 0.788 for BERT, 0.766 for DistilBERT, and 0.77 for RoBERTa.

## Dataset
The dataset consists of 5147 book titles and their descriptions, labeled with genres such as:
•⁠  ⁠Science Fiction/Fantasy
•⁠  ⁠Mystery/Thriller
•⁠  ⁠Romance
•⁠  ⁠Literature
•⁠  ⁠History
•⁠  ⁠Psychology
•⁠  ⁠And more...

## Models Used
•⁠  ⁠*BERT (Base Uncased)*: A transformer model pre-trained on a large corpus of English data.
•⁠  ⁠*RoBERTa*: A robustly optimized BERT pre-training approach with modified hyperparameters.
•⁠  ⁠*DistilBERT*: A smaller, faster version of BERT, distilled from the original model.

## Training Process
•⁠  ⁠*Tokenization*: Data was tokenized using model-specific tokenizers.
•⁠  ⁠*Fine-Tuning*: Models were fine-tuned with the following hyperparameters:
  - Learning Rate: 2e-5
  - Epochs: 4
  - Batch Size: 16

## Results
The models achieved the following accuracies:
•⁠  ⁠*BERT*: 78.8%
•⁠  ⁠*DistilBERT*: 76.6%
•⁠  ⁠*RoBERTa*: 77.0%

## Challenges
•⁠  ⁠*Data Collection*: Ethical issues with scraping data from Goodreads led us to use alternative sources.
•⁠  ⁠*Dataset Size*: The dataset was relatively small, which impacted generalization. This was mitigated by supplementing with a Kaggle dataset.
•⁠  ⁠*Ambiguous Genres*: Some genres were hard to classify, leading to misclassifications.

## Future Work
•⁠  ⁠Expand the dataset with more diverse book summaries.
•⁠  ⁠Explore additional models like GPT-based architectures.
•⁠  ⁠Address class imbalances using data augmentation techniques.


## References
•⁠  ⁠[books.toscrape.com](https://books.toscrape.com/index.html)
•⁠  ⁠[Kaggle Dataset](https://www.kaggle.com/datasets/athu1105/book-genre-prediction/data?select-data.csv)
•⁠  ⁠[Hugging Face BERT](https://huggingface.co/google-bert/bert-base-uncased#model-description)
•⁠  ⁠[Hugging Face RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta)
•⁠  ⁠[Hugging Face DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)

## Conclusion
We successfully fine-tuned multiple pre-trained models for book genre classification, achieving good accuracy. BERT performed slightly better, but DistilBERT offers a good balance between accuracy and speed. This project demonstrates the potential of NLP in enhancing the book discovery process for readers.
