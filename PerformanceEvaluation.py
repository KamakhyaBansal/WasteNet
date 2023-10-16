def get_iou(x,y):
    intersection = np.logical_and(x, y)
    union = np.logical_or(x,y)
    iou_score.append(np.sum(intersection) / np.sum(union))

from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score


iou_score = []
precision = []
recall = []
batch_size = 4
index = 615
fileslist = []

labels = [0, 1]
jaccards_list = []

for X, Y in tqdm(val_loader, total=len(val_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        Y_pred = model(X)
        for i in range(test_batch_size):

          jaccards = []

          out = np.asarray(torch.argmax(Y_pred[i].cpu(),dim=0).unsqueeze(2).repeat(1,1,3).reshape(-1, 3).cpu())
          out *=255
          label_pred = label_model1.predict(out).reshape(256, 256)
          pred = Image.fromarray(label_pred*255)
          #actual = Y[i].cpu().numpy().reshape(256, 256)
          #act = Image.fromarray(actual.astype(np.uint8)*255)
          #pred.save("/content/drive/MyDrive/SatelliteImages/Results/UNet/"+str(index)+".png")

          Y[i][Y[i]!=0] = 255
          Y[i][Y[i]==0] = 1
          Y[i][Y[i]==255] = 0

          for label in labels:
            jaccard = jaccard_score(label_pred.flatten(),Y[i].cpu().numpy().flatten(), pos_label=label)
            jaccards.append(jaccard)
          jaccards_list.append(np.mean(jaccards))
          index+=1

          #print(get_iou(act,pred))
          #iou_score.append(get_iou(act,pred))
          precision.append(precision_score(label_pred,Y[i].cpu().numpy(),average='micro'))
          recall.append(recall_score(label_pred,Y[i].cpu().numpy(),average='micro'))



sum(jaccards_list)/len(jaccards_list)

sum(precision)/len(precision)

sum(recall)/len(recall)

def get_map(x,iou_score):
    fp = 0
    tp = 0
    for i in range(len(iou_score)):
        if iou_score[i]>=x:
            tp+=1
        else:
            fp+=1
    return(tp/(tp+fp))

#MAP[0.5:0.95] step of 0.05


map=[]
iouList = [0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]
for i in range(len(iouList)):
    map.append(get_map(iouList[i],jaccards_list))

m1 = map[iouList.index(0.5)]
m2 = sum(map)/len(map)


print("M1: ",m1)
print("M2: ",m2)
map
