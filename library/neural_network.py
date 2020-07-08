import os

ask_base_dir=input("What is your base extracted zip file directory?")
train_dir=os.path.join(ask_base_dir,"train")
validation_dir = os.path.join(ask_base_dir,"validation")

ask_directory_folder=input("How many training variable are there?")

for i in range(ask_directory_folder):
    
