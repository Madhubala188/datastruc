age=int(input("Enter the age:"))
weight=int(input("Enter the weight:"))
def medicine():
    if age >= 15 and age < 18 and weight >= 55:
        print("You should take the medicine.")
    else:
        print("You do not need to take the medicine.")
medicine()
