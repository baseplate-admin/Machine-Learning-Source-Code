ask_zip=input("Do you want to extract ZIP?")
if ask_zip == 'Yes' or ask_zip=='yes':
    from library.zip_extract import zip_extract
    zip_extract()
