from pickle import dump, load

def save_object(obj, filename):
    """Save an object to a file
    Parameters
    ----------
    obj: object
        The object to save.
    filename: str
        The path of the file to save the object in.
    """
    with open(filename, "wb") as out_file:
        dump(obj, out_file)

def load_object(filename):
    """Load an object from a file.
    Parameters
    ---------
    filename: str
        The path of the input file.
    Returns
    -------
    obj: object
        The loaded object.
    """
    with open(filename, "rb") as in_file:
        return load(in_file)

def misclassification_rate(ytrue, ypred):
    """The rate of samples which are misclassified
    Parameters
    ----------
    ytrue: sequence of objects
        The true labels
    ypred: sequence of objects
        The predicted labels
    Returns
    -------
    rate: float
        The error rate.
    """
    if len(ytrue) != len(ypred):
        raise ValueError("ytrue and ypred must have the same length")
    return sum(1.0 if t != p else 0.0 for t, p in zip(ytrue, ypred)) / len(ytrue)