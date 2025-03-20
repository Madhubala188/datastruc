mark=int(input("Enter the marks:"))
def grade_calculator():

    if mark>=90:
      print("It scored Grade A")
    elif mark>=80:
        print("It scored Grade B")
    elif mark>=70:
        print("It scored Grade c")
    elif mark>=60:
        print("It scored Grade D")
    else:
        print("Grade E")

grade_calculator()