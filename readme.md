

## 标注形式设置

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  <KeyPointLabels name="KeyPointLabels" toName="image">
    <Label value="text" smart="true" background="#e51515" showInline="true"/>
  </KeyPointLabels>
    <RectangleLabels name="RectangleLabels" toName="image">
  	<Label value="text" background="#0d14d3"/>
  </RectangleLabels>
  <PolygonLabels name="PolygonLabels" toName="image">
    <Label value="text" background="rgba(255, 0, 0, 0.7)"/>
  </PolygonLabels>
    <BrushLabels name="BrushLabels" toName="image">
  	<Label value="cat" background="#FF0000"/>
  	<Label value="person" background="#0d14d3"/>
  </BrushLabels>
</View>
```

## 启动label-studio 前端

```shell
label-studio -p 9081 
```

## 启动自动标注后台服务

```shell
label-studio-ml start tke-detector --with \
config_file=tke-detector/config/r50_tt.yaml \
checkpoint_file=tke_model_final.pth
```


## 启动交互式标注后台服务

```shell
label-studio-ml start sam --port 8003 --with \
  model_name=mobile_sam  \
  sam_config=vit_t \
  sam_checkpoint_file=./TT_mbsam_785.pt \
  out_mask=True \
  out_bbox=True \
 #device=cuda:0 
device=cpu 
```

## [创建项目](docs/创建项目.gif)

![创建项目](https://github.com/TankosTao/Label-Studio-for-Text/tree/master/docs/创建项目.gif)


## [启动深度学习后端服务](docs/启动服务.gif)

![启动服务](https://github.com/TankosTao/Label-Studio-for-Text/tree/master/docs/启动服务.gif)

## [自动标注和交互式标注](docs/自动标注和交互式标注.gif)

![自动标注和交互式标注](https://github.com/TankosTao/Label-Studio-for-Text/tree/master/docs/自动标注和交互式标注.gif)