def main():
    print("1/sqrt(2) (|hello> + |goodbye>) world!")


if __name__ == "__main__":
    from .circuit import Circuit
    c = Circuit(2)
    main()
