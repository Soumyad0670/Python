def display_menu():
    """Display available list operations"""
    print("\nList Operations Menu:")
    print("1. Add element")
    print("2. Remove element")
    print("3. Insert element at position")
    print("4. Sort list")
    print("5. Reverse list")
    print("6. Find element")
    print("7. Count element occurrences")
    print("8. Display list")
    print("9. Clear list")
    print("0. Exit")

def perform_list_operations():
    try:
        user_list = input("Enter elements separated by space: ").split()
        user_list = [int(x) if x.isdigit() else x for x in user_list]
    except ValueError:
        print("Invalid input! Please try again.")
        return
    while True:
        display_menu()
        choice = input("Enter your choice (0-9): ")
        if choice == '0':
            print("Exiting program...")
            break
        elif choice == '1':
            element = input("Enter element to add: ")
            user_list.append(element if not element.isdigit() else int(element))
            print(f"Added {element} to list")
        elif choice == '2':
            if not user_list:
                print("List is empty!")
                continue
            element = input("Enter element to remove: ")
            try:
                user_list.remove(element if not element.isdigit() else int(element))
                print(f"Removed {element} from list")
            except ValueError:
                print("Element not found in list!")
        elif choice == '3':
            try:
                pos = int(input("Enter position: "))
                element = input("Enter element: ")
                user_list.insert(pos, element if not element.isdigit() else int(element))
                print(f"Inserted {element} at position {pos}")
            except ValueError:
                print("Invalid position!")
        elif choice == '4':
            try:
                user_list.sort()
                print("List sorted!")
            except TypeError:
                print("Cannot sort list with mixed types!")
        elif choice == '5':
            user_list.reverse()
            print("List reversed!")
        elif choice == '6':
            element = input("Enter element to find: ")
            try:
                pos = user_list.index(element if not element.isdigit() else int(element))
                print(f"Element found at position {pos}")
            except ValueError:
                print("Element not found in list!")
        elif choice == '7':
            element = input("Enter element to count: ")
            count = user_list.count(element if not element.isdigit() else int(element))
            print(f"Element occurs {count} times")
        elif choice == '8':
            print("\nCurrent list:", user_list)

        elif choice == '9':
            user_list.clear()
            print("List cleared!")

        else:
            print("Invalid choice! Please try again.")

        print("\nCurrent list:", user_list)

if __name__ == "__main__":
    perform_list_operations()