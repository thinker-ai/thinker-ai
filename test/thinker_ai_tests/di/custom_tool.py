from thinker_ai.agent.roles.di.data_interpreter import DataInterpreter
from thinker_ai_tests.di.my_custom_tool import magic_function


async def main():
    di = DataInterpreter(tools=["magic_function"])
    await di.run("Just call the magic function with arg1 'A' and arg2 2. Tell me the result.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
