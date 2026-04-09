from Src.data_loader import load_data, validate_data

path = 'heart.csv'
def main():

    df = load_data(path)
    df = validate_data(df)

if __name__ == "__main__":
    main()

