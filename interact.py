from amiya import Amiya

amiya = Amiya()


info = """
========================================================
#            Wellcome to use Amiya
========================================================
"""
print(info)

while True:
    raw_input = input("[User]:")
    if raw_input == "exit":
        break
    response = amiya.get_response(text=raw_input)
    print("[Amiya]:{response}".format(response=response))
