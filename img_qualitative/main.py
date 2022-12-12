import json
import os

if __name__ == '__main__':

    if not os.path.exists("mega_json"):
        os.mkdir("mega_json")

    with open("3pic.json", "r") as source_json_fp:
        source_json = json.load(source_json_fp)

        for image in source_json['images']:
            new_json = {'images': [image]}
            for key in source_json:
                if key == "images":
                    continue
                new_json[key] = source_json[key]

            filename = str(image['file']).split('.')[0]
            print("Writing to {}.json.".format(filename))

            with open("mega_json/{}.json".format(filename), "w") as new_json_fp:
                json.dump(new_json, new_json_fp, indent=1)
