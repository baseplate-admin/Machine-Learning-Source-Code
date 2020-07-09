
ask_zip=input("Do you want to extract ZIP?")
if ask_zip == 'Yes' or ask_zip=='yes':
    from library.zip_extract import zip_extract
    zip_extract()
elif ask_zip=="no" or ask_zip=="No":
    print("We are skipping ZIP extraction!")
from library.neural_network import model
ask_number=input("""
What is your neural network number?
1.Default
""")
ask_train_dir=input("What is your Train Directory?")
ask_validation_dir=input("What is your Validation Directory?")
model(ask_number,ask_train_dir,ask_validation_dir)
