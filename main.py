import json

def main():
    data = None
    with open('train.json') as json_file:
        data = json.load(json_file)
    N = len(data)
    #for i in range(0, N):
    i = 500
    imgId = data[i]['imgId']
    objIds = data[i]['objId']
    objNames = data[i]['objName']
    bboxes = data[i]['bboxes']
    print(imgId)
    print(objIds)
    print(objNames)
    print(bboxes)

if __name__ == '__main__':
    main()