from multi import Multi

def main():
    multi = Multi()
    multi.add_profile(
        profile_name="additiong test",
        profile_description="addition test profile",
        tools=[
            ("addition", "addition tool", """
            def addition(a, b):
                return a+b
            """)
        ]
    )
    print(multi.get_profiles())
    # print(multi.get_profile(profile_id="a319519b-29da-426b-9e27-933431444070"))
    # multi.add_llm(
    #     base_url="https://api.research.computer/v1",
    #     api_key="abc",
    #     model_name="Qwen/Qwen2.5-7B-Instruct-1M",
    #     llm_name="basic_llm"
    # )
    # print(multi.get_llms())


if __name__ == "__main__":
    main() 