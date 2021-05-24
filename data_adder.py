
with open('data/wrong.csv', encoding='utf-8') as adder:
    contents = adder.readlines()

    with open('data/updated_dataset.csv', 'a', encoding='utf-8') as file:
        for content in contents:
            file.write(content)

with open('data/wrong.csv', 'w', encoding='utf-8') as empty_file:
    empty_file.write('')
