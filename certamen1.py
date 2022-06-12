numero_bultos=0
peso_avion=0
costo=0
ingreso=0
numero_equipaje=0
suma=0
equipajes=[]
while peso_avion<=2000:
  peso = int(input("peso de equipaje: "))
  if peso <=500:
    peso_avion=peso_avion+peso
    if peso_avion<=2000:
        if peso<=500:
          if peso<=25:
            equipajes.append(peso)
            costo=peso*1000
            ingreso=costo+ingreso
            numero_equipaje=numero_equipaje+1
            suma=peso+suma
            print("hola su costo es de",costo,"pesos")
          elif peso<=300:
            equipajes.append(peso)
            costo=peso*1500
            ingreso=costo+ingreso
            numero_equipaje=numero_equipaje+1
            suma=peso+suma
            print("hola su costo es de",costo,"pesos")
          else :
            equipajes.append(peso)
            costo=peso*2000
            ingreso=costo+ingreso
            numero_equipaje=numero_equipaje+1
            suma=peso+suma
            print("hola su costo es de",costo,"pesos")
    else :
        print("el peso ya a sido sobrepasado no se aceptan mas equipajes")
  else :
        print("hola el peso de equipaje no debe superar los 500kg porfavor ingrese otro equipaje con menor peso80")
  

peso_avion = peso_avion-peso
promedio = suma/numero_equipaje
bulto_mas_pesado = max(equipajes)

print("el ingreso total :",ingreso,"pesos")
print("el peso total es de : ",peso_avion,"kg")
print("total de equipajes: ",numero_equipaje)
print("promedio: ",promedio,"kg")
print("peso de los equipajes: ",equipajes,"kg")
print("bulto mas pesado : ",bulto_mas_pesado,"kg")
