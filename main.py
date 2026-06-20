from user import User

user_count = 0
users = {}

user = None

def new_user():
    global user_count
    users[user_count] = User()
    user_count += 1

def login(id: int):
    global user
    user = users[id]

def add_password(application: str, path: str) -> str:
    return user.add_password(application, path)

def check_password(application: str, path: str) -> str:
    return user.check_password(application, path)

if __name__ == "__main__":
    import os
    _dir = os.path.dirname(os.path.abspath(__file__))
    new_user()
    login(0)
    print(add_password("Netflix", os.path.join(_dir, "mug1.jpeg")))
    print(check_password("Netflix", os.path.join(_dir, "mug2.jpeg")))