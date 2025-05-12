import bcrypt

# Choose a password for your test user
password_to_hash = b"password123"  # b"" makes it a byte string, which bcrypt needs

# Generate the salt and hash the password
hashed_password_bytes = bcrypt.hashpw(password_to_hash, bcrypt.gensalt())

# Decode the byte string to a regular string to save in JSON
hashed_password_str = hashed_password_bytes.decode('utf-8')

print(f"Username: testuser")
print(f"Password (for your reference, don't save this directly): {password_to_hash.decode('utf-8')}")
print(f"Hashed Password (for users.json): {hashed_password_str}")