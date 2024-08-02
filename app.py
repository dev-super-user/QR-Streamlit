import cv2
import numpy as np
import streamlit as st
from pyzbar import pyzbar
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd

# Definir los productos con campos adicionales
def create_product_dictionary():
    product_dict = {
        "7501055310883": {
            "Descripción": "Agua Embotellada Ciel 1L: Agua purificada de manantial, fresca y cristalina, en una práctica botella de 1 litro. Ideal para hidratarte durante el día con un sabor natural y sin aditivos. Perfecta para llevar a cualquier lugar.",
            "Especialidad": "Oncología",
            "Usuario": "Voluntario 0A",
            "Zona": "Puebla",
            "Departamento": "3ED",
            "Código de Almacén": "ALM001"
        },
        "234567890123": {
            "Descripción": "Ibuprofeno 200mg - 20 tabletas",
            "Especialidad": "Medicamentos",
            "Usuario": "Voluntario B",
            "Zona": "Veracruz",
            "Departamento": "7UJ",
            "Código de Almacén": "ALM002"
        },
        "345678901234": {
            "Descripción": "Paracetamol 500mg - 50 tabletas",
            "Especialidad": "Medicamentos",
            "Usuario": "Voluntario C",
            "Zona": "Tlaxcala",
            "Departamento": "B12",
            "Código de Almacén": "ALM003"
        },
        "456789012345": {
            "Descripción": "Vitamina C 1000mg - 30 cápsulas",
            "Especialidad": "Suplementos",
            "Usuario": "Voluntario D",
            "Zona": "Puebla",
            "Departamento": "A12",
            "Código de Almacén": "ALM004"
        },
        # Añade aquí más productos según sea necesario
    }
    return product_dict

product_dictionary = create_product_dictionary()

detections = []
last_detection_times = {}

def decode(frame):
    decoded_objects = pyzbar.decode(frame)
    detected = False  
    
    for obj in decoded_objects:
        barcode_data = obj.data.decode("utf-8")
        product_info = product_dictionary.get(barcode_data, None)
        if product_info:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            timestamp_dt = datetime.now()

            if (barcode_data not in last_detection_times or 
                timestamp_dt - last_detection_times[barcode_data] > timedelta(seconds=1)):
                
                detections.append({
                    "Fecha de lectura": timestamp,
                    "Código": barcode_data,
                    "Descripción": product_info["Descripción"],
                    "Especialidad": product_info["Especialidad"],
                    "Usuario": product_info["Usuario"],
                    "Zona": product_info["Zona"],
                    "Departamento": product_info["Departamento"],
                    "Código de Almacén": product_info["Código de Almacén"]
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
            cv2.putText(frame, product_info["Descripción"], (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
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
            st.error("Error: No se pudo capturar el frame.")
            break
        
        frame, detected = decode(frame)
        
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
