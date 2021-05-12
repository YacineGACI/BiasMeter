import csv

base_filename = "corpora/SBIC.v2/SBIC.v2.{}.{}"
subset = "trn"


# Open the origial CSV file
with open(base_filename.format(subset, "csv"), 'r') as f:
    csv_content = csv.reader(f)

    # Open the new to be generated txt file
    with open(base_filename.format(subset, 'txt'), 'w') as g:

        header = True
        for line in csv_content:
            if header:
                header = False
                continue

            target_is_group = float(line[0]) if line[0] != '' else 0.0 # 0 if the target of the post is a person, 1 if it is a group (social stereotype)

            post = line[14].replace('\n', '')
            stereotype_summary = line[17].replace('\n', '')

            new_entry = post + ". " + stereotype_summary + "\n"

            # Only keep posts and sentences that are targetted at groups, and remove those that are targetted at individuals
            if target_is_group:
                g.write(new_entry)