from kerneldensityestimator import KDE

def main():
    """Example usage of KDE class."""

    kde = KDE()
    print(f"Bandwidth (h) = {kde.bandwidth}")

    kde.training_model.load("train.csv")
    accuracy = kde.get_training_accuracy()
    print(f"Training Accuracy {accuracy}%")

    kde.testing_model.load("test.csv")
    accuracy = kde.get_testing_accuracy()
    print(f"Testing Accuracy {accuracy}%")

    # Warning, this function can be slow....
    kde.plot_class_boundaries(kde.testing_model)


    kde.training_model.load("zipcode_train.csv")
    accuracy = kde.get_training_accuracy()
    print(f"Training Accuracy {accuracy}%")

    kde.testing_model.load("zipcode_test.csv")
    accuracy = kde.get_testing_accuracy()
    print(f"Testing Accuracy {accuracy}%")

    #We can't plot this data, it is 16D, not 2D.

if __name__ == "__main__":
    main()
