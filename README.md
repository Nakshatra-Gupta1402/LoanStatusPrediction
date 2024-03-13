# LoanStatusPrediction

Abstract

This project focuses on predicting loan status using three different classification algorithms:
Logistic Regression, K-Nearest Neighbors (KNN), and Naive Bayes. The objective is to determine
whether a loan application will be approved or denied based on certain features provided by the
applicants

Data Set
The dataset used for this project is sourced from Kaggle: Loan Status Prediction Dataset:
Loan_Prediction_dataset . It comprises information related to loan applications, including
applicant demographics, financial information, and credit history. Each application is labeled
with its corresponding loan status, indicating whether it was approved or denied. Data
preprocessing techniques are employed to clean and standardize the data before training and
testing the models. Features extracted from the dataset include applicant income,
loan amount, credit score, and employment status.

Problem Definition
1. The project aims to develop a loan status prediction model to determine whether a loan
   application will be approved or denied based on applicant information. Leveraging
   machine learning algorithms like Logistic Regression, K-Nearest Neighbors (KNN), and
   Naive Bayes, the objective is to accurately classify loan applications into approved or
   denied categories.
2. The dataset comprises a collection of loan applications, each annotated with its loan
   status label. Features extracted from the dataset, including applicant demographics,
   financial information, and credit history, are used to train the models. The task involves
   preprocessing the data, splitting it into training and testing sets, and evaluating the
   performance of the classification algorithms.
3. The project addresses the need for automated loan approval systems to streamline the
   lending process and improve efficiency in financial institutions.

Solution Methodology
● Logistic Regression: A binary classification algorithm that predicts the probability of a
patient experiencing heart failure based on input medical features. Logistic Regression is
interpretable and efficient, making it suitable for heart failure prediction tasks. However,
it assumes linear relationships between features and may not capture complex patterns
in the data.2
● K-Nearest Neighbors (KNN): A simple yet effective classification algorithm that classifies
data points based on the majority class of their nearest neighbors in the feature space.
KNN works well with non-linear data and doesn't make strong assumptions about the
underlying data distribution. However, its performance can be sensitive to the choice of k
and may require careful preprocessing of the data to normalize feature scales.
● Naive Bayes: A probabilistic classification algorithm based on Bayes' theorem with the
"naive" assumption of feature independence. Naive Bayes calculates the probability of a
patient experiencing heart failure given their medical features and selects the class with
the highest probability as the prediction. Despite its simple assumption, Naive Bayes can
be surprisingly effective, especially with high-dimensional data. It's computationally
efficient and works well with small datasets. However, the assumption of feature
independence may not hold in real-world scenarios, impacting its accuracy.

Programming Tools
● Python (Jupyter IDE)
● Python Libraries : Seaborn, Matplot (for visualisation) ; Numpy , Pandas and sklearn.

Data Source
The dataset used for this project is sourced from Kaggle, specifically from the following link:
Loan Status Prediction Dataset. In this Loan Status Prediction dataset, we have the data of
applicants who previously applied for the loan based on the property, which is a Property Loan.
The bank will decide whether to give a loan to the applicant based on some factors such as
Applicant Income, Loan Amount, previous Credit History, Co-applicant Income, etc. The goal is to
build a Machine Learning Model to predict the loan to be approved or to be rejected for an
applicant.
About the loan_data.csv file:
● Loan_ID: A unique loan ID.
● Gender: Either male or female.
● Married: Whether Married (yes) or Not Married (No).
● Dependents: Number of persons depending on the client.
● Education: Applicant Education (Graduate or Undergraduate).
● Self_Employed: Self-employed (Yes/No).
● ApplicantIncome: Applicant income.3
● CoapplicantIncome: Co-applicant income.
● LoanAmount: Loan amount in thousands.
● Loan_Amount_Term: Terms of the loan in months.
● Credit_History: Credit history meets guidelines.
● Property_Area: Applicants are living either Urban, Semi-Urban or Rural.
● Loan_Status: Loan approved (Y/N).
