from dataloader import load_high

datapath = '../../anonymized_patient_notes.txt'
essay = load_high(datapath)

medr = essay.split("history of present illness:")[1:]


for i, pie in enumerate(medr):
    file_path = "./dataset/report_" + str(i) + '.txt'
    with open(file_path, 'w') as file:
        file.write(pie)