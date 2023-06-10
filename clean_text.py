import os

docs = os.listdir("texts")
# remove line breaks for each file
for i in docs:
    with open(f"texts/{i}", "r", encoding="utf-8") as f:
        with open(f"texts/clean_{i}", "w", encoding="utf-8") as f2:
            lines = f.readlines()

            for line in lines:
                if "Session" not in line:
                    if len(line) > 2:
                        if line[-2] != " ":
                            f2.write(line.replace("\n", " "))
                        else:
                            f2.write(line.replace("\n", ""))
                    else:
                        f2.write(line.replace("\n", ""))
