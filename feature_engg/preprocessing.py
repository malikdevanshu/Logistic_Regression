"""preprocessing.py 
   Handles all data preprocessing steps:
   scaling the data 
   Spliting the data into train test
 """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def splitting(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=test_size, random_state=random_state)
    print(f"\n Shape after Splitting")
    print(f"\nX_train:{X_train.shape} | X_test:{X_test.shape} \n y_train:{y_train.shape} | y_test:{y_test.shape}")
    return (X_train, X_test, y_train, y_test)

def scale_features(X_train, X_test):
    model_scal = StandardScaler()
    X_train_scaled = model_scal.fit_transform(X_train)
    X_test_scaled = model_scal.transform(X_test)
    return (X_train_scaled, X_test_scaled, model_scal)
