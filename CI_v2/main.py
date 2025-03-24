from interpreter import execute_task, initialize_agent, load_environment


def main():
    """Main function to interact with the user and execute the request."""
    print("🚀 Welcome to the AI Python Generator!")

    user_request = input(
        "Enter what you want to generate (e.g., 'Create a QR code', 'Generate a Fibonacci sequence'): "
    )

    print("\n🔄 Setting up the environment...")
    load_environment()

    print("🤖 Initializing AI agent...")
    agent = initialize_agent()

    print("\n📝 Processing your request...")
    try:
        response = execute_task(agent, user_request)
        print("\n✅ Task Completed!\n")
        print(response)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
