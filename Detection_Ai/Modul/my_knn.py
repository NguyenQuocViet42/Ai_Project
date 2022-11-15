import math

class my_knn:
    def __init__(self):
        pass
    
    def calcDistancs(self, pointA, pointB, numOfFeature=2):
        tmp = 0
        for i in range(numOfFeature):
            tmp += (float(pointA[i]) - float(pointB[i])) ** 2
        return math.sqrt(tmp)
    
    def kNearestNeighbor(self,trainSet, Y, point, k):
        distances = []
        for i in range(len(trainSet)):
            distances.append({
                "label": Y[i],
                "value": self.calcDistancs(trainSet[i,:], point)
            })
        distances.sort(key=lambda x: x["value"])
        labels = [item["label"] for item in distances]
        return labels[:k]
    
    def find(self, arr, k):
        labels = set(arr) # set label
        ans = ""
        maxOccur = 0
        for label in labels:
            num = arr.count(label)
            if num > maxOccur:
                maxOccur = num
                ans = label
        return ans, maxOccur/k *100
    
    def predict(self,trainSet, Y, point, k):
        knn = self.kNearestNeighbor(trainSet, Y, point, k)
        answer = self.find(knn, k)
        return answer