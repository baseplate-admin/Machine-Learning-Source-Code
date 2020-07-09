def zip_extract():
    import os
    from zipfile import ZipFile

    def zip_function():
        print("We are extracting ZIP!!!")
        where_is_zip=input("What is your zip location?")
        what_is_zip_name=input("What is your zip name?")
        what_is_zip_extension=input("What is your ZIP format?")
        zip_join=os.path.join(where_is_zip,what_is_zip_name+ '.'+ what_is_zip_extension)
        with ZipFile(zip_join,"r") as zip:
            zip.extractall()
            zip.printdir()
            print("Enter a Number or It will cause ValueError.")
    how_many_zip=int(input('How many zip do you want to extract?'))
    try:
        print("""
        This is a number!!
        Lets Go!!!
        """)
        for  i in range(how_many_zip):
            ask_if_zip_extract=input("""
            Do you want to extract zip?
            Enter 0 to skip extracting zip.
            Enter 1 to to extract ZIP.
            """)
            if int(ask_if_zip_extract)==0:
                zip_function(2)
            elif int(ask_if_zip_extract)==1:
                zip_function(1)
            else:
                print("Theres a problem with zip extract.")
    except Exception as e:
        print(e)
