# VisionAVI Annotation Instruction

## labels.txt
```bash
angryface
auto 
bicycle 
bus 
car 
happyface 
motorcycle 
neutralface 
numberplate 
person 
pole 
road 
sadface 
sidewalk 
truck
van
road_cross
```

## Instruction : 

- pole  means : any pole on street, it can have street light, traffic light or anything. all will be considered as pole
- truck : Big or small will be considered as truck
- car : omni van, car will be considered as car
- bus : government bus, privat bus or simila looking ones will be considered as bus
- bicycle : electric bicycles, bicycle or similar looking ones are considered as bicycle
- auto : auto rickshaw of any color will be auto
- van : loading vehicles, ambulance, large pickup, food van's van but not "omni"

- Instruction for face, only if you are more than 70% confident about a person emotion, then use happyface or neutralface or sadface. If you are confused about person face emotion and you confidence is below 70% then mark them neutralface.

- Person : Toddler, young, adult will be considered as person

- Sidewalk : only if it is clear that there is side walk or footpath, then annotate sidewalk else leave it.

- road : If side walk is there but there is no distinction between road and sidewalk like in the image then mark all of it as a road.

    * If road has lot of vehicles and road is not visible, then don't mark the road.

    * If big portion of the road is visible, then only mark it.

- road_cross : Includes zebra cross or yellow road cross
