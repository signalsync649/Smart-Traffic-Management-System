from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from datetime import datetime
import pytz

app = FastAPI()

# Database connection
conn = psycopg2.connect(
    dbname="Trafficdb",
    user="postgres",
    password="Shivanshks@2006",
    host="localhost",
    port=5432
)
conn.autocommit = True

class TrafficData(BaseModel):
    signal_id: int
    timestamp: str
    hour_slot: str
    total_vehicles: int
    vehicles: dict

@app.post("/traffic")
def receive_traffic(data: TrafficData):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO traffic_counts (signal_id, timestamp, hour_slot, total_vehicles, cars, bikes, buses, trucks, others, emergency_clearances)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                data.signal_id,
                data.timestamp,
                data.hour_slot,
                data.total_vehicles,
                data.vehicles.get("car", 0),
                data.vehicles.get("motorbike", 0),
                data.vehicles.get("bus", 0),
                data.vehicles.get("truck", 0),
                data.vehicles.get("others", 0),
                data.vehicles.get("emergency", 0)
            ))
        return {"status": "success"}
    except Exception as e:
        print("Database error:", e)
        raise HTTPException(status_code=500, detail="Database error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
