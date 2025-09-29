import express from "express";
import session from "express-session";
import bodyParser from "body-parser";
import path from "path";
import fs from "fs";
import ejs from "ejs";
import { fileURLToPath } from "url";
import pkg from "pg";
import dayjs from "dayjs";
import http from "http";
import { Server } from "socket.io";
import utc from "dayjs/plugin/utc.js";
import timezone from "dayjs/plugin/timezone.js";

// Load plugins
dayjs.extend(utc);
dayjs.extend(timezone);
dayjs.tz.setDefault("Asia/Kolkata");

const { Pool, types } = pkg;

/* ---------- Parse PG numeric types as JS numbers ---------- */
types.setTypeParser(20, (val) => (val === null ? null : parseInt(val, 10)));
types.setTypeParser(1700, (val) => (val === null ? null : parseFloat(val)));

// Database connection
const pool = new Pool({
  user: "postgres",
  host: "localhost",
  database: "Trafficdb",
  password: "Shivanshks@2006",
  port: 5432
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Create HTTP server and attach Socket.IO
const server = http.createServer(app);
const io = new Server(server);

/* ---------- Middleware ---------- */
app.use(bodyParser.urlencoded({ extended: true }));
app.use(session({
  secret: "supersecretkey",
  resave: false,
  saveUninitialized: true
}));
app.use(express.static(path.join(__dirname, "public")));

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

/* ---------- Auth ---------- */
const USER = { username: "admin", password: "1234" };

function isAuthenticated(req, res, next) {
  if (req.session.user) return next();
  res.redirect("/login");
}

/* ---------- Render helper ---------- */
function renderWithLayout(res, view, options = {}) {
  try {
    const viewPath = path.join(__dirname, "views", view + ".ejs");
    const body = ejs.render(
      fs.readFileSync(viewPath, "utf8"),
      options,
      { filename: viewPath }
    );
    res.render("layout", {
      showSidebar: options.showSidebar ?? true,
      ...options,
      body
    });
  } catch (e) {
    console.error("Render error:", e);
    res.status(500).send("Template render error");
  }
}

/* ========================= ROUTES ========================= */

// Home page
app.get("/", isAuthenticated, async (req, res) => {
  try {
    const result = await pool.query("SELECT area_id, area_name FROM areas");
    renderWithLayout(res, "welcome", {
      title: "Welcome - Traffic Management",
      user: req.session.user,
      areas: result.rows,
      showSidebar: false
    });
  } catch (err) {
    console.error(err);
    res.status(500).send("Error fetching areas");
  }
});

// Area selection
app.post("/select-area", isAuthenticated, (req, res) => {
  const { area_id } = req.body;
  req.session.selectedArea = area_id || null;
  res.redirect("/dashboard");
});

// Dashboard (today in IST)
app.get("/dashboard", isAuthenticated, async (req, res) => {
  if (!req.session.selectedArea) return res.redirect("/");
  const areaId = req.session.selectedArea;

  try {
    // Define "today" in IST
    const todayWhere = `
      (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date = (now() AT TIME ZONE 'Asia/Kolkata')::date
    `;

    // 1️⃣ Hourly totals
    const hourlyResult = await pool.query(
      `
      WITH hours AS (SELECT generate_series(0, 23) AS hour)
      SELECT 
        LPAD(h.hour::text, 2, '0') || ':00-' || LPAD((h.hour + 1)::text, 2, '0') AS hour_slot,
        COALESCE(SUM(tc.total_vehicles), 0) AS total_vehicles
      FROM hours h
      LEFT JOIN traffic_counts tc
        ON EXTRACT(HOUR FROM (tc.timestamp AT TIME ZONE 'Asia/Kolkata')) = h.hour
        AND ${todayWhere}
        AND tc.signal_id IN (SELECT signal_id FROM signals WHERE area_id = $1)
      GROUP BY h.hour
      ORDER BY h.hour
      `,
      [areaId]
    );

    const labels = hourlyResult.rows.map(r => r.hour_slot);
    const values = hourlyResult.rows.map(r => Number(r.total_vehicles || 0));

    // 2️⃣ Vehicle breakdown
    const dailyTotals = await pool.query(
      `
      SELECT
        COALESCE(SUM(tc.cars), 0)   AS cars,
        COALESCE(SUM(tc.bikes), 0)  AS bikes,
        COALESCE(SUM(tc.buses), 0)  AS buses,
        COALESCE(SUM(tc.trucks), 0) AS trucks,
        COALESCE(SUM(tc.others), 0) AS others
      FROM traffic_counts tc
      WHERE tc.signal_id IN (SELECT signal_id FROM signals WHERE area_id = $1)
        AND ${todayWhere}
      `,
      [areaId]
    );
    const vehicleBreakdown = dailyTotals.rows[0] || { cars: 0, bikes: 0, buses: 0, trucks: 0, others: 0 };

    // 3️⃣ Stats cards
    const stats = await pool.query(
      `
      SELECT 
        COALESCE(SUM(tc.total_vehicles), 0) AS total_vehicles,
        COALESCE(SUM(tc.emergency_clearances), 0) AS emergency_clearances
      FROM traffic_counts tc
      WHERE tc.signal_id IN (SELECT signal_id FROM signals WHERE area_id = $1)
        AND ${todayWhere}
      `,
      [areaId]
    );
    const totalVehiclesToday = Number(stats.rows[0]?.total_vehicles || 0);
    const emergencyClearances = Number(stats.rows[0]?.emergency_clearances || 0);

    // 4️⃣ Peak hours
    const peakResult = await pool.query(
      `
  WITH hours AS (SELECT generate_series(0, 23) AS hour)
  SELECT 
    LPAD(h.hour::text, 2, '0') || ':00-' || LPAD((h.hour + 1)::text, 2, '0') AS hour_slot,
    COALESCE(SUM(tc.total_vehicles), 0) AS total
  FROM hours h
  LEFT JOIN traffic_counts tc
    ON EXTRACT(HOUR FROM (tc.timestamp AT TIME ZONE 'Asia/Kolkata')) = h.hour
    AND ${todayWhere}
    AND tc.signal_id IN (SELECT signal_id FROM signals WHERE area_id = $1)
  GROUP BY h.hour
  ORDER BY total DESC
  LIMIT 5
  `,
      [areaId]
    );

    const peakLabels = peakResult.rows.map(r => r.hour_slot);
    const peakValues = peakResult.rows.map(r => Number(r.total));

    // 5️⃣ Weekly trends
    const weeklyResult = await pool.query(
      `
      WITH days AS (
        SELECT generate_series(
          date_trunc('day', now() AT TIME ZONE 'Asia/Kolkata') - interval '6 days',
          date_trunc('day', now() AT TIME ZONE 'Asia/Kolkata'),
          interval '1 day'
        )::date AS day
      )
      SELECT 
        to_char(d.day, 'YYYY-MM-DD') AS day_key,
        to_char(d.day, 'Dy') AS label,
        COALESCE(SUM(tc.total_vehicles), 0) AS total
      FROM days d
      LEFT JOIN traffic_counts tc
        ON (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date = d.day
        AND tc.signal_id IN (SELECT signal_id FROM signals WHERE area_id = $1)
      GROUP BY d.day
      ORDER BY d.day
      `,
      [areaId]
    );

    const weeklySlots = weeklyResult.rows.map(r => ({
      key: r.day_key,
      label: r.label.trim(),
      total: Number(r.total || 0)
    }));

    const todayDate = dayjs().tz("Asia/Kolkata").format("YYYY-MM-DD");

    renderWithLayout(res, "dashboard", {
      title: "Dashboard - Traffic Management",
      user: req.session.user,
      showSidebar: true,
      areaId,
      labels,
      values,
      vehicleBreakdown,
      totalVehiclesToday,
      emergencyClearances,
      todayDate,
      peakLabels,
      peakValues,
      weeklySlots,
      weeklyLabels: weeklySlots.map(s => s.label),
      weeklyValues: weeklySlots.map(s => s.total),
      from: req.query.from || "",
      to: req.query.to || ""
    });

  } catch (err) {
    console.error("Dashboard error:", err);
    res.status(500).send("Error fetching dashboard data");
  }
});

/* ========================= ANALYTICS ========================= */

app.get("/analytics", isAuthenticated, async (req, res) => {
  const areaId = req.session.selectedArea;
  if (!areaId) return res.redirect("/");
  try {
    const fromDate = req.query.from
      ? dayjs(req.query.from).tz("Asia/Kolkata").format("YYYY-MM-DD")
      : dayjs().tz("Asia/Kolkata").subtract(6, "day").format("YYYY-MM-DD");

    const toDate = req.query.to
      ? dayjs(req.query.to).tz("Asia/Kolkata").format("YYYY-MM-DD")
      : dayjs().tz("Asia/Kolkata").format("YYYY-MM-DD");

    const today = dayjs().tz("Asia/Kolkata").format("YYYY-MM-DD");
    const last7 = dayjs().tz("Asia/Kolkata").subtract(6, "day").format("YYYY-MM-DD");
    const last30 = dayjs().tz("Asia/Kolkata").subtract(29, "day").format("YYYY-MM-DD");

    const periodLabel = (fromDate === toDate)
      ? `on ${fromDate}`
      : `from ${fromDate} to ${toDate}`;

    // 1️⃣ Hourly breakdown (signal-wise traffic)
    const hourlyBreakdown = await pool.query(
      `
  SELECT 
    to_char((tc.timestamp AT TIME ZONE 'Asia/Kolkata'), 'HH24:00') AS hour_slot,
    s.signal_name,
    COALESCE(SUM(tc.cars), 0)   AS cars,
    COALESCE(SUM(tc.bikes), 0)  AS bikes,
    COALESCE(SUM(tc.buses), 0)  AS buses,
    COALESCE(SUM(tc.trucks), 0) AS trucks,
    COALESCE(SUM(tc.others), 0) AS others,
    COALESCE(SUM(tc.total_vehicles), 0) AS total
  FROM traffic_counts tc
  JOIN signals s ON tc.signal_id = s.signal_id
  WHERE s.area_id = $1
    AND (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date BETWEEN $2::date AND $3::date
  GROUP BY to_char((tc.timestamp AT TIME ZONE 'Asia/Kolkata'), 'HH24:00'), s.signal_name
  ORDER BY hour_slot, s.signal_name
  `,
      [areaId, fromDate, toDate]
    );

    // 2️⃣ Vehicle type share (donut chart)
    const vehicleTypeShare = await pool.query(
      `
  SELECT 'cars' AS type,   COALESCE(SUM(tc.cars), 0)   AS count
  FROM traffic_counts tc JOIN signals s ON tc.signal_id = s.signal_id
  WHERE s.area_id = $1 AND (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date BETWEEN $2 AND $3
  UNION ALL
  SELECT 'bikes',  COALESCE(SUM(tc.bikes), 0)
  FROM traffic_counts tc JOIN signals s ON tc.signal_id = s.signal_id
  WHERE s.area_id = $1 AND (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date BETWEEN $2 AND $3
  UNION ALL
  SELECT 'buses',  COALESCE(SUM(tc.buses), 0)
  FROM traffic_counts tc JOIN signals s ON tc.signal_id = s.signal_id
  WHERE s.area_id = $1 AND (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date BETWEEN $2 AND $3
  UNION ALL
  SELECT 'trucks', COALESCE(SUM(tc.trucks), 0)
  FROM traffic_counts tc JOIN signals s ON tc.signal_id = s.signal_id
  WHERE s.area_id = $1 AND (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date BETWEEN $2 AND $3
  UNION ALL
  SELECT 'others', COALESCE(SUM(tc.others), 0)
  FROM traffic_counts tc JOIN signals s ON tc.signal_id = s.signal_id
  WHERE s.area_id = $1 AND (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date BETWEEN $2 AND $3
  `,
      [areaId, fromDate, toDate]
    );

    // 3️⃣ Daily traffic trend (✅ fixed GROUP BY)
    const dailyTrend = await pool.query(
      `
  SELECT 
    to_char((tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date, 'YYYY-MM-DD') AS day,
    COALESCE(SUM(tc.total_vehicles), 0) AS total
  FROM traffic_counts tc
  JOIN signals s ON tc.signal_id = s.signal_id
  WHERE s.area_id = $1
    AND (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date BETWEEN $2 AND $3
  GROUP BY (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date
  ORDER BY day
  `,
      [areaId, fromDate, toDate]
    );

    // 4️⃣ Signal comparison (busiest signals)
    const signalComparison = await pool.query(
      `
  SELECT 
    s.signal_name,
    COALESCE(SUM(tc.total_vehicles), 0) AS total
  FROM traffic_counts tc
  JOIN signals s ON tc.signal_id = s.signal_id
  WHERE s.area_id = $1
    AND (tc.timestamp AT TIME ZONE 'Asia/Kolkata')::date BETWEEN $2 AND $3
  GROUP BY s.signal_name
  ORDER BY total DESC
  `,
      [areaId, fromDate, toDate]
    );


    const topSignal = signalComparison.rows.length
      ? { name: signalComparison.rows[0].signal_name, total: signalComparison.rows[0].total }
      : null;

    // 🔹 Format signal trends
    const grouped = {};
    hourlyBreakdown.rows.forEach(r => {
      if (!grouped[r.signal_name]) {
        grouped[r.signal_name] = { signal: r.signal_name, hours: [], values: [] };
      }
      grouped[r.signal_name].hours.push(r.hour_slot);
      grouped[r.signal_name].values.push(Number(r.total));
    });
    const signalTrends = Object.values(grouped);

    // 🔹 Vehicle types hourly
    const vehicleTypes = { hours: [], cars: [], bikes: [], buses: [], trucks: [], others: [] };
    const hourlyMap = {};
    hourlyBreakdown.rows.forEach(r => {
      if (!hourlyMap[r.hour_slot]) {
        hourlyMap[r.hour_slot] = { cars: 0, bikes: 0, buses: 0, trucks: 0, others: 0 };
      }
      hourlyMap[r.hour_slot].cars += Number(r.cars);
      hourlyMap[r.hour_slot].bikes += Number(r.bikes);
      hourlyMap[r.hour_slot].buses += Number(r.buses);
      hourlyMap[r.hour_slot].trucks += Number(r.trucks);
      hourlyMap[r.hour_slot].others += Number(r.others);
    });
    Object.keys(hourlyMap).sort().forEach(hour => {
      vehicleTypes.hours.push(hour);
      vehicleTypes.cars.push(hourlyMap[hour].cars);
      vehicleTypes.bikes.push(hourlyMap[hour].bikes);
      vehicleTypes.buses.push(hourlyMap[hour].buses);
      vehicleTypes.trucks.push(hourlyMap[hour].trucks);
      vehicleTypes.others.push(hourlyMap[hour].others);
    });

    const dailyTrendData = dailyTrend.rows.map(d => ({
      date: d.day,
      total: Number(d.total)
    }));

    renderWithLayout(res, "analytics", {
      title: "Analytics - Traffic Management",
      user: req.session.user,
      showSidebar: true,
      areaId,
      from: fromDate,
      to: toDate,
      today,
      last7,
      last30,
      periodLabel,
      hourlyBreakdown: hourlyBreakdown.rows,
      vehicleTypeShare: vehicleTypeShare.rows,
      dailyTrend: dailyTrendData,
      signalComparison: signalComparison.rows,
      topSignal,
      signalTrends,
      vehicleTypes
    });

  } catch (err) {
    console.error("Analytics error:", err.message, err.stack);
    res.status(500).send("Error fetching analytics data");
  }
});

/* ========================= CHALLANS ========================= */
app.get("/challans", isAuthenticated, async (req, res) => {
  const areaId = req.session.selectedArea;
  if (!areaId) return res.redirect("/");

  try {
    // Get query params
    const { from, to } = req.query;

    // Calculate date ranges for quick buttons
    const today = new Date();
    const last7 = new Date();
    last7.setDate(today.getDate() - 6); // Last 7 days includes today
    const last30 = new Date();
    last30.setDate(today.getDate() - 29); // Last 30 days includes today

    // Format dates as YYYY-MM-DD for input fields
    const formatDate = (d) => d.toISOString().split("T")[0];

    // Build WHERE clause dynamically
    const whereClauses = ["area_id = $1"];
    const params = [areaId];
    let paramIndex = 2;

    if (from) {
      whereClauses.push(`violation_time::date >= $${paramIndex}`);
      params.push(from);
      paramIndex++;
    }

    if (to) {
      whereClauses.push(`violation_time::date <= $${paramIndex}`);
      params.push(to);
      paramIndex++;
    }

    const query = `
      SELECT *
      FROM challan_details_view
      WHERE ${whereClauses.join(" AND ")}
      ORDER BY violation_time DESC
      LIMIT 100
    `;

    const result = await pool.query(query, params);

    renderWithLayout(res, "challan", {
      title: "E-Challans - Traffic Management",
      user: req.session.user,
      showSidebar: true,
      challans: result.rows,
      areaId,
      from: from || "",
      to: to || "",
      today: formatDate(today),
      last7: formatDate(last7),
      last30: formatDate(last30)
    });
  } catch (err) {
    console.error("❌ Challan fetch error:", err);
    res.status(500).send("Error fetching challans");
  }
});


/* ========================= ACCIDENTS ========================= */
app.get("/accidents", isAuthenticated, async (req, res) => {
  const areaId = req.session.selectedArea;
  if (!areaId) return res.redirect("/");

  // Get from/to dates from query params
  const from = req.query.from ? new Date(req.query.from) : null;
  const to = req.query.to ? new Date(req.query.to) : null;

  try {
    let query = `
      SELECT *
      FROM accident_details_view
      WHERE area_id = $1
    `;
    const params = [areaId];

    if (from && to) {
      query += " AND accident_time::date BETWEEN $2 AND $3";
      params.push(from.toISOString().split("T")[0], to.toISOString().split("T")[0]);
    }

    query += " ORDER BY accident_time DESC LIMIT 100";

    const result = await pool.query(query, params);

    // Dates for quick buttons
    const today = new Date().toISOString().split("T")[0];
    const last7 = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split("T")[0];
    const last30 = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

    renderWithLayout(res, "accidents", {
      title: "Accidents - Traffic Management",
      user: req.session.user,
      showSidebar: true,
      accidents: result.rows,
      areaId,
      from: req.query.from || today,
      to: req.query.to || today,
      today,
      last7,
      last30
    });

  } catch (err) {
    console.error("❌ Accident fetch error:", err);
    res.status(500).send("Error fetching accidents");
  }
});




/* ========================= AUTH ========================= */
app.get("/login", (req, res) => {
  if (req.session.user) return res.redirect("/");
  renderWithLayout(res, "login", {
    title: "Login",
    error: null,
    user: null,
    showSidebar: false
  });
});

app.post("/login", (req, res) => {
  const { username, password } = req.body;
  if (username === USER.username && password === USER.password) {
    req.session.user = { username };
    return res.redirect("/");
  }
  renderWithLayout(res, "login", {
    title: "Login",
    error: "Invalid username or password",
    user: null,
    showSidebar: false
  });
});

app.get("/logout", (req, res) => {
  req.session.destroy(() => res.redirect("/login"));
});

/* ========================= SOCKET.IO + PG LISTENER ========================= */
io.on("connection", (socket) => {
  console.log("🔌 Client connected:", socket.id);

  socket.on("disconnect", () => {
    console.log("❌ Client disconnected:", socket.id);
  });
});

async function initPgListener() {
  const client = await pool.connect();

  client.on("error", (err) => {
    console.error("❌ PG listener error:", err);
    setTimeout(initPgListener, 5000); // retry on error
  });

  client.on("end", () => {
    console.warn("⚠️ PG listener disconnected, retrying...");
    setTimeout(initPgListener, 5000); // retry on disconnect
  });

  client.on("notification", (msg) => {
    try {
      const data = JSON.parse(msg.payload);

      // Format timestamp consistently
      if (data.violation_time) {
        data.formattedTime = dayjs(data.violation_time)
          .tz("Asia/Kolkata")
          .format("YYYY-MM-DD HH:mm:ss");
      }
      if (data.accident_time) {
        data.formattedTime = dayjs(data.accident_time)
          .tz("Asia/Kolkata")
          .format("YYYY-MM-DD HH:mm:ss");
      }

      switch (msg.channel) {
        case "traffic_update":
          data.date = dayjs(data.timestamp).tz("Asia/Kolkata").format("YYYY-MM-DD");
          data.hour = data.hour_slot || null;
          io.emit("traffic_update", data);
          break;

        case "challan_update":
          io.emit("challan_update", data);
          break;

        case "accident_update":
          io.emit("accident_update", data);
          break;

        default:
          console.warn("⚠️ Unknown PG channel:", msg.channel);
      }

      console.log(`📡 DB notification [${msg.channel}]:`, data);

    } catch (err) {
      console.error("❌ Notification parse error:", err);
    }
  });

  // Listen to all required channels
  await client.query("LISTEN traffic_update");
  await client.query("LISTEN challan_update");
  await client.query("LISTEN accident_update");
  console.log("👂 Listening for PG channels: traffic_update, challan_update, accident_update");
}

// Start the PostgreSQL listener with auto-retry
initPgListener().catch((err) => {
  console.error("❌ Failed to start PG listener:", err);
  setTimeout(initPgListener, 5000);
});

/* ========================= SERVER ========================= */
server.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
});
