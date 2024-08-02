import cv2
import numpy as np
import streamlit as st
from pyzbar.pyzbar import decode  # Cambia esto a pyzbar.pyzbar
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd

def create_product_dictionary():
    product_dict = {
        "7501055310883": {
            "Descripción": "Agua Embotellada Ciel 1L",
            "Especialidad": "Hidratación",
            "Usuario": "General",
            "Zona": "Nacional",
            "Departamento": "Bebidas",
            "Código de almacen": "001"
        },
        "234567890123": {
            "Descripción": "Ibuprofeno 200mg - 20 tabletas",
            "Especialidad": "Medicamentos",
            "Usuario": "Adultos",
            "Zona": "Nacional",
            "Departamento": "Farmacia",
            "Código de almacen": "002"
        },
        # Agrega más productos aquí
    }
    return product_dict

product_dictionary = create_product_dictionary()

detections = []
last_detection_times = {}

def decode_frame(frame):
    decoded_objects = decode(frame)
    detected = False  
    
    for obj in decoded_objects:
        barcode_data = obj.data.decode("utf-8")
        product_info = product_dictionary.get(barcode_data, {})
        description = product_info.get("Descripción", "Descripción no disponible")
        especialidad = product_info.get("Especialidad", "N/A")
        usuario = product_info.get("Usuario", "N/A")
        zona = product_info.get("Zona", "N/A")
        departamento = product_info.get("Departamento", "N/A")
        almacen = product_info.get("Código de almacen", "N/A")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_dt = datetime.now()

        if (barcode_data not in last_detection_times or 
            timestamp_dt - last_detection_times[barcode_data] > timedelta(seconds=1)):
            
            detections.append({
                "Fecha de lectura": timestamp,
                "Código": barcode_data,
                "Descripción": description,
                "Especialidad": especialidad,
                "Usuario": usuario,
                "Zona": zona,
                "Departamento": departamento,
                "Código de almacen": almacen
            })
            
            last_detection_times[barcode_data] = timestamp_dt

            detected = True  # Se ha detectado un código

        points = obj.polygon
        if len(points) > 4: 
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points
            
        n = len(hull)
        for j in range(0, n):
            pt1 = (int(hull[j][0]), int(hull[j][1]))
            pt2 = (int(hull[(j + 1) % n][0]), int(hull[(j + 1) % n][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
            
        x = obj.rect.left
        y = obj.rect.top
        cv2.putText(frame, f'Código: {barcode_data}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, description, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, detected

def main():
    st.title("Deep Axiom - Lector de Códigos de Barras y QR")

    st.subheader("Historial de Detecciones")
    detections_df = pd.DataFrame(detections)  
    stframe = st.empty()
    table = st.empty()

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: No se pudo abrir la cámara.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, detected = decode_frame(frame)
        
        if detected:
            # Actualizar la tabla de detecciones
            detections_df = pd.DataFrame(detections)
            table.write(detections_df)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        stframe.image(image, channels="RGB", use_column_width=True)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
