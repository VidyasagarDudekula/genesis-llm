import tiktoken
with open('input.txt', 'r') as file:
    shake_sp_data = file.read()
    shake_length = len(shake_sp_data)
with open('grimms.txt', 'r') as file:
    grimms_data = file.read()
    grimms_length = len(grimms_data)
with open('sherlock.txt', 'r') as file:
    sherlocks_data = file.read()
    sherlocks_length = len(sherlocks_data)
with open('dracula.txt', 'r') as file:
    dracula_data = file.read()
    dracula_length = len(dracula_data)

total_text_data = f"""
Poems:-

{shake_sp_data[:int(shake_length*0.9)]}


END of Poems



stories:-

{grimms_data[:int(grimms_length*0.9)]}

END of Stories


Sherlocks stories:-

{sherlocks_data[:int(sherlocks_length*0.9)]}

End of Sherlocks stories.


Dracula Stories:-

{dracula_data[:int(dracula_length*0.9)]}

"""

complete_text_data = f"""
Poems:-

{shake_sp_data}


END of Poems


stories:-

{grimms_data}

END of Stories


Sherlocks stories:-

{sherlocks_data}

End of Sherlocks stories.


Dracula Stories:-

{dracula_data}

"""

validation_data = f"""
Poems:-

{shake_sp_data[int(shake_length*0.9):]}


END of Poems


stories:-

{grimms_data[int(grimms_length*0.9):]}

END of Stories


Sherlocks stories:-

{sherlocks_data[int(sherlocks_length*0.9):]}

End of Sherlocks stories.


Dracula Stories:-

{dracula_data[int(dracula_length*0.9):]}

"""