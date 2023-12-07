import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import streamlit as st

# Load the configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
# Generate hashed passwords
hashed_passwords = stauth.Hasher(['Password@1098', 'Password@1097', 'Password@109']).generate()

# Replace plain text passwords with hashed passwords
for user, password in zip(config['credentials']['usernames'].keys(), hashed_passwords):
    config['credentials']['usernames'][user]['password'] = password

# Save the updated configuration file
with open('config.yaml', 'w') as file:
    yaml.safe_dump(config, file)

# Render the login module
name, authentication_status, username = authenticator.login('Login', 'main')

# Authenticating users Use the returned name and authentication_status to allow verified users to access restricted content. Optionally, add a logout button
if authentication_status:
    authenticator.logout('Logout', 'main', key='unique_key')
    st.write(f'Welcome *{name}*')
    st.title('Some content')
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')

st.link_button("google","https://www.google.co.in/")

# Creating a password reset widget Allow logged-in users to modify their password using the reset_password widget
if authentication_status:
    try:
        if authenticator.reset_password(username, 'Reset password'):
            st.success('Password modified successfully')
    except Exception as e:
        st.error(e)

# Create a new user registration widget Allow users to sign up for your application using the register_user widget
try:
    if authenticator.register_user('Register New User', preauthorization=False):
        st.success('User registered successfully')
except Exception as e:
    st.error(e)
    # Remember to update the configuration file after using this widget.


# Create a forgot password widget Allow users to generate a new random password using the forgot_password widget
try:
    username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
    if username_forgot_pw:
        st.success('New password sent securely')
        # Random password to be transferred to the user securely
    else:
        st.error('Username not found')
except Exception as e:
    st.error(e)

# Create a forgot username widget Allow users to retrieve their forgotten username using the forgot_username widget
try:
    username_forgot_username, email_forgot_username = authenticator.forgot_username('Forgot username')
    if username_forgot_username:
        st.success('Username sent securely')
        # Username to be transferred to the user securely
    else:
        st.error('Email not found')
except Exception as e:
    st.error(e)

# Create an update user details widget Allow logged-in users to update their name and/or email using the update_user_details widget
if authentication_status:
    try:
        if authenticator.update_user_details(username, 'Update user details'):
            st.success('Entries updated successfully')
    except Exception as e:
        st.error(e)
        # Remember to update the configuration file after using this widget

#  Update the configuration file Ensure that the configuration file is saved whenever the credentials are updated or any of the password reset, user registration, forgot password, or update user details widgets are used
with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)







