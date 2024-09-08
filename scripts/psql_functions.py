from mail_functions import retrieve_mails

from config import *

import psycopg2
from datetime import datetime

def psql_cursor(dbname: str, user: str, password: str, host: str, port: int):
    """
    Establishes a connection to a PostgreSQL database and returns the connection and cursor objects.

    Args:
        dbname (str): The name of the database.
        user (str): The database user.
        password (str): The password for the database user.
        host (str): The host address of the database.
        port (int): The port number on which the database server is listening.

    Returns:
        tuple: A tuple containing the connection and cursor objects.
    """
    conn = psycopg2.connect(
        dbname = dbname,
        user = user,
        password = password,
        host = host,
        port = port
    )
    cursor = conn.cursor()

    return conn,cursor


def user_exists(cursor,username):
    """
    Checks if a user exists in the 'user_activity' table.

    Args:
        cursor: The cursor object for executing SQL queries.
        username (str): The username to check.

    Returns:
        bool: True if the user exists, False otherwise.
    """
    check_user_query = """
    SELECT COUNT(*) FROM user_activity WHERE "user" = %s
    """
    cursor.execute(check_user_query, (username,))
    user_exists = cursor.fetchone()[0] > 0

    return user_exists


def log_user_activity(conn,cursor,username: str):
    """
    Logs or updates the last activity time for a user in the 'user_activity' table.

    Args:
        conn: The database connection object.
        cursor: The cursor object for executing SQL queries.
        username (str): The username to log or update.

    Returns:
        None
    """
    current_time = datetime.now()

    if user_exists(cursor, username):
        update_query = """
        UPDATE user_activity
        SET last_fetch = %s
        WHERE user = %s
        """
        cursor.execute(update_query, (current_time, username))
        print(f"{current_time}:\tLast fetch updated for user: {username}")

    else:
            
        insert_query = """
        INSERT INTO user_activity
        VALUES (%s, %s)
        """
        cursor.execute(insert_query, (username, current_time))
        print(f"{current_time}:\tNew user {username} logged")

    # Commit the transaction
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

def get_last_fetch(cursor,username: str):
    """
    Retrieves the last fetch time for a user from the 'user_activity' table.

    Args:
        cursor: The cursor object for executing SQL queries.
        username (str): The username for which to retrieve the last fetch time.

    Returns:
        datetime: The last fetch time for the user.
    """
    fetch_query = """
        SELECT "last_fetch"
        FROM user_activity
        WHERE "user" = %s
        """
    cursor.execute(fetch_query, (username,))
    last_fetch = cursor.fetchone()[0]

    return last_fetch

def main():
    conn, cursor = psql_cursor(dbname = PSQL_DB_NAME,
                            user = PSQL_USER,
                            password = PSQL_PASSWORD,
                            host = PSQL_HOST,
                            port = PSQL_PORT)

    print(get_last_fetch(cursor,'manav'))


if __name__ == "__main__":
    main()