import sqlite3

DATABASE = "users.db"


def register_user(name, email, password):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    try:

        cursor.execute(

            """
            INSERT INTO users(name,email,password)
            VALUES(?,?,?)
            """,

            (name,email,password)

        )

        conn.commit()

        return True

    except:

        return False

    finally:

        conn.close()


def login_user(email,password):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute(

        """
        SELECT * FROM users
        WHERE email=? AND password=?
        """,

        (email,password)

    )

    user = cursor.fetchone()

    conn.close()

    return user