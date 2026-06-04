import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


OUT = Path("BaoCao_TienDo_Tuan1_XGB_DQN.docx")
WHULX_METRICS = Path("artifacts/outputs/whulx_reproduction_metrics.json")
WHULX_RESULT = Path("artifacts/outputs/whulx_reproduction/result.json")


def load_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def fmt_num(value, digits=2, suffix=""):
    if value is None:
        return "Chưa có"
    return f"{float(value):.{digits}f}{suffix}"


def run(text, bold=False, size=24):
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:r>"
        f"<w:rPr>{bold_xml}<w:rFonts w:ascii=\"Arial\" w:hAnsi=\"Arial\" "
        f"w:eastAsia=\"Arial\" w:cs=\"Arial\"/><w:sz w:val=\"{size}\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{escape(str(text))}</w:t>"
        "</w:r>"
    )


def p(text="", style=None):
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    return f"<w:p><w:pPr>{style_xml}</w:pPr>{run(text)}</w:p>"


def heading(text, level=1):
    return p(text, f"Heading{level}")


def bullet(text):
    return f'<w:p><w:pPr><w:pStyle w:val="ListBullet"/></w:pPr>{run(text)}</w:p>'


def table(headers, rows):
    def cell(text, bold=False):
        return (
            "<w:tc><w:tcPr><w:tcW w:w=\"4500\" w:type=\"dxa\"/></w:tcPr>"
            "<w:p>"
            + run(text, bold=bold, size=22)
            + "</w:p></w:tc>"
        )

    parts = [
        "<w:tbl>",
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "</w:tblBorders></w:tblPr>",
        "<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>",
    ]
    for row in rows:
        parts.append("<w:tr>" + "".join(cell(c) for c in row) + "</w:tr>")
    parts.append("</w:tbl>")
    return "".join(parts)


metrics = load_json(WHULX_METRICS)
result = load_json(WHULX_RESULT)
if result:
    metrics = {"raw_shape": result["raw_shape"], "after_clean_shape": result["after_clean_shape"], **result["xgboost"]}

body = []

body.append(heading("BÁO CÁO TIẾN ĐỘ TUẦN 1 - KHẢO SÁT XGB-DQN", 1))
body.append(p("Chủ đề: Khảo sát repo WHU-LX/Hvac-Window-based-XGB-DQN và định hướng triển khai mô phỏng HVAC bằng DQN."))
body.append(p("Nguồn khảo sát chính: https://github.com/WHU-LX/Hvac-Window-based-XGB-DQN"))

body.append(heading("1. Mục tiêu tuần 1", 2))
body.append(p("Tuần 1 tập trung khảo sát repo gốc WHU-LX/Hvac-Window-based-XGB-DQN để hiểu bài toán, dataset, thuật toán và cách đánh giá kết quả. Repo này đi kèm nghiên cứu Occupant-centric HVAC and window control: A reinforcement learning model for enhancing indoor thermal comfort and energy efficiency, Building and Environment, 2024."))
body.append(p("Mục tiêu của nhóm là tái hiện được pipeline gốc ở mức có thể kiểm chứng, sau đó dùng kết quả khảo sát để xây dựng hướng mô phỏng riêng bằng Sinergym/EnergyPlus cho khí hậu Việt Nam."))

body.append(heading("2. Keyword và phạm vi khảo sát", 2))
body.append(table(
    ["Keyword", "Ý nghĩa trong đề tài"],
    [
        ["HVAC and window control", "Điều khiển đồng thời hệ HVAC và trạng thái cửa sổ để tác động đến môi trường trong phòng."],
        ["Occupant behavior", "Hành vi người dùng ảnh hưởng đến điều hòa, cửa sổ, nhiệt độ phòng và tiện nghi."],
        ["Reinforcement learning", "DQN học chính sách chọn action dựa trên state hiện tại và reward nhận được."],
        ["Thermal comfort", "Tiện nghi nhiệt là mục tiêu chính, được đánh giá qua vùng nhiệt độ tiện nghi theo ASHRAE adaptive comfort."],
        ["Energy efficiency", "Chính sách điều khiển cần giảm mức dùng điều hòa nhưng không làm giảm tiện nghi quá mức."],
        ["XGBoost + DQN", "XGBoost dự đoán chuyển trạng thái nhiệt độ, DQN chọn action điều khiển."],
    ],
))

body.append(heading("3. Dataset trong repo gốc", 2))
body.append(p("Repo WHU-LX sử dụng dữ liệu dạng CSV đã xử lý, gồm Cleaned_data.csv và Cleaned_data_encode.csv. Đây là dữ liệu occupant behavior, không phải dữ liệu mô phỏng EnergyPlus trực tiếp. Dữ liệu ghi lại trạng thái điều hòa, cửa sổ, nhiệt độ, độ ẩm, thời tiết và các biến thời gian để học quan hệ giữa hành động điều khiển và nhiệt độ trong phòng ở bước tiếp theo."))
body.append(p("Các nhóm cột chính gồm: AC_Status, Window_Status, CLast_Time, WLast_Time, Indoor_Temp, Indoor_RH, Outdoor_Temp, Outdoor_RH, Rain, Cloud, Windspeed, Month, Hour, Room_ID, Study_ID, City, Next_Indoor_Temp, Next_Indoor_RH, Next_Outdoor_Temp, Next_Outdoor_RH, Differ_Outdoor_Temp, Differ_Indoor_Temp và Target_Temp."))
body.append(p("Trong notebook gốc, DQN dùng num_features = 8. State được lấy từ 8 biến đầu của data_test:"))
for item in ["Indoor_Temp", "Indoor_RH", "Outdoor_Temp", "Outdoor_RH", "Rain", "Cloud", "Windspeed", "Hour"]:
    body.append(bullet(item))
body.append(p("Việc đưa Rain, Cloud và Windspeed vào state phù hợp với bài toán gốc vì mô hình có điều khiển cửa sổ. Các biến thời tiết phụ này ảnh hưởng trực tiếp đến quyết định mở cửa và trao đổi nhiệt giữa trong nhà và ngoài trời."))

body.append(heading("4. Thuật toán và cấu hình trong repo gốc", 2))
body.append(p("Repo gốc dùng hướng XGB-DQN. XGBoost được dùng để dự đoán Differ_Indoor_Temp, tức độ thay đổi nhiệt độ trong nhà ở timestep tiếp theo. Sau đó DQN dùng mô hình XGBoost như môi trường chuyển trạng thái để thử action, tính reward và học chính sách điều khiển."))
body.append(table(
    ["Thành phần", "Giá trị trong repo gốc"],
    [
        ["num_features", "8"],
        ["num_actions", "24"],
        ["Mạng Q", "2 hidden layer Dense(64, ReLU), output linear theo 24 action"],
        ["gamma", "0.9"],
        ["epsilon khởi tạo", "1.0"],
        ["min_epsilon", "0.1"],
        ["epsilon_decay", "0.995"],
        ["learning_rate", "0.001"],
        ["memory_capacity", "10000"],
        ["batch_size", "32"],
        ["num_episodes", "1000"],
    ],
))
body.append(p("Action trong repo gốc gồm 24 lựa chọn. Action 0 là AC off và window closed. Action 1-11 bật AC, đóng cửa sổ và đặt Target_Temp = 19 + action, tương ứng 20-30 C. Action 12 là AC off và window open. Các action lớn hơn 12 kết hợp bật AC và mở cửa sổ, với target temperature khoảng 20-30 C."))
body.append(p("Reward gốc dựa trên vùng tiện nghi ASHRAE adaptive comfort. Nếu indoor temperature nằm trong vùng tiện nghi thì reward không bị phạt. Nếu nhiệt độ thấp hơn hoặc cao hơn vùng tiện nghi, reward bị phạt theo bình phương khoảng cách. Ngoài ra, action dùng AC bị trừ chi phí năng lượng; action vừa dùng AC vừa mở cửa bị phạt nặng hơn."))

body.append(heading("5. Kết quả tái hiện pipeline gốc", 2))
body.append(p("Phần này viết theo hướng so sánh của tài liệu tham khảo: nêu mục tiêu gốc, benchmark dùng để đối chiếu, bảng metric và nhận xét. Cần phân biệt hai loại kết quả: kết quả công bố trong README/bài báo gốc và kết quả nhóm đã chạy lại trong môi trường hiện tại."))
body.append(heading("5.1 Đối chiếu với mục tiêu gốc", 3))
body.append(table(
    ["KPI", "Kết quả repo/bài báo gốc", "Tình trạng tái hiện"],
    [
        ["Comfort duration", "Tăng 24% thời lượng tiện nghi nhiệt.", "Đã ghi nhận làm mục tiêu tham chiếu; chưa tái hiện toàn bộ nhiều ngày/nhiều thành phố như bài báo."],
        ["AC usage / energy", "Giảm 24.7% mức sử dụng điều hòa.", "Đã ghi nhận làm mục tiêu tham chiếu; pipeline hiện tái hiện một ngày mẫu từ notebook."],
        ["XGBoost transition model", "Dự đoán Differ_Indoor_Temp để mô phỏng next indoor temperature.", "Đã tái hiện được bằng Cleaned_data.csv."],
        ["DQN policy", "Học chính sách HVAC + window control.", "Đã chạy lại 1000 episodes cho một ngày mẫu theo logic notebook gốc."],
    ],
))

body.append(heading("5.2 Kết quả XGBoost tái hiện", 3))
if metrics:
    body.append(table(
        ["Metric", "Giá trị"],
        [
            ["Dữ liệu gốc", f"{metrics['raw_shape'][0]:,} dòng x {metrics['raw_shape'][1]} cột"],
            ["Sau làm sạch", f"{metrics['after_clean_shape'][0]:,} dòng x {metrics['after_clean_shape'][1]} cột"],
            ["Input XGBoost", f"{metrics['x_shape'][0]:,} dòng x {metrics['x_shape'][1]} biến"],
            ["Train/Test", f"{metrics['train_shape'][0]:,} train / {metrics['test_shape'][0]:,} test"],
            ["Target", "Differ_Indoor_Temp"],
            ["MAE", f"{metrics['mae']:.4f} C"],
            ["RMSE", f"{metrics['rmse']:.4f} C"],
            ["R2", f"{metrics['r2']:.4f}"],
        ],
    ))
    body.append(p("MAE khoảng 0.18 C và RMSE khoảng 0.35 C cho thấy phần mô hình chuyển trạng thái có thể dự đoán biến đổi nhiệt độ trong nhà ở mức đủ ổn để dùng làm môi trường gần đúng cho DQN. R2 khoảng 0.48 nghĩa là mô hình giải thích được một phần đáng kể biến thiên của target, nhưng vẫn chưa phải mô hình vật lý hoàn chỉnh."))
else:
    body.append(p("Chưa có file metric XGBoost để chèn vào báo cáo."))

body.append(heading("5.3 Kết quả DQN tái hiện một ngày mẫu", 3))
if result:
    dqn = result["dqn"]
    human = result["human_baseline"]
    action_dist = ", ".join(f"action {k}: {v:.1f}%" for k, v in dqn["action_distribution_pct"].items())
    body.append(table(
        ["Metric", "Human baseline", "DQN tái hiện"],
        [
            ["Comfort trong khung 6-17h", fmt_num(human["comfort_pct"], 2, "%"), fmt_num(dqn["comfort_pct"], 2, "%")],
            ["Mean indoor temperature", fmt_num(human["mean_indoor_temp"], 2, " C"), fmt_num(dqn["mean_indoor_temp"], 2, " C")],
            ["AC on ratio", "Theo dữ liệu người dùng", fmt_num(dqn["ac_on_pct"], 2, "%")],
            ["Window open ratio", "Theo dữ liệu người dùng", fmt_num(dqn["window_open_pct"], 2, "%")],
            ["Total reward", "-", fmt_num(dqn["total_reward"], 2)],
            ["Action distribution", "-", action_dist],
        ],
    ))
    body.append(p(f"Pipeline DQN đã chạy lại {result['episodes']} episodes theo cấu hình gốc. Với ngày mẫu đầu tiên trong dataset, human baseline đạt {human['comfort_pct']:.2f}% comfort và DQN tái hiện đạt {dqn['comfort_pct']:.2f}% comfort trong khung điều khiển 6-17h. Chính sách học được chủ yếu chọn các action không dùng điều hòa, với phân bố {action_dist}. Vì AC on ratio bằng {dqn['ac_on_pct']:.2f}%, chi phí điều hòa trong ngày mẫu này bằng 0 theo reward đang dùng."))
    body.append(p("Kết quả này xác nhận pipeline XGB-DQN có thể chạy lại, nhưng chưa đủ để kết luận đã tái hiện toàn bộ kết quả bài báo. Lý do là notebook gốc kiểm tra trên một ngày cụ thể, còn kết quả +24% comfort duration và -24.7% AC usage trong README/bài báo là kết quả tổng hợp trên phạm vi rộng hơn. Do đó báo cáo ghi rõ đây là kết quả tái hiện một ngày mẫu, dùng để kiểm chứng logic thuật toán trước khi mở rộng đánh giá."))

    evaluation = result.get("evaluation")
    if evaluation:
        body.append(heading("5.4 Kết quả đánh giá mở rộng trên nhiều ngày", 3))
        rows = []
        for item in evaluation["controllers"]:
            rows.append(
                [
                    item["controller"],
                    fmt_num(item.get("comfort_pct"), 2, "%"),
                    fmt_num(item.get("ac_on_pct"), 2, "%"),
                    fmt_num(item.get("window_open_pct"), 2, "%"),
                    fmt_num(item.get("mean_indoor_temp"), 2, " C"),
                    fmt_num(item.get("total_reward"), 2),
                ]
            )
        body.append(table(
            ["Controller", "Comfort", "AC on", "Window open", "Mean indoor", "Total reward"],
            rows,
        ))
        dqn_eval = next((item for item in evaluation["controllers"] if item["controller"] == "DQN"), None)
        human_eval = next((item for item in evaluation["controllers"] if item["controller"] == "Human"), None)
        window_eval = next((item for item in evaluation["controllers"] if item["controller"] == "Window_Open"), None)
        if dqn_eval and human_eval:
            body.append(p(f"Khi đánh giá trên {evaluation['eval_days']} ngày hoàn chỉnh, DQN đạt {dqn_eval['comfort_pct']:.2f}% comfort, thấp hơn human baseline {human_eval['comfort_pct']:.2f}%, nhưng không bật điều hòa trong toàn bộ rollout. Điều này cho thấy reward hiện tại đang ưu tiên tránh chi phí AC rất mạnh, nên chính sách học được thiên về action 0 và action 12."))
        if dqn_eval and window_eval:
            body.append(p(f"So với baseline luôn mở cửa sổ, DQN có comfort {dqn_eval['comfort_pct']:.2f}% so với {window_eval['comfort_pct']:.2f}% và mở cửa sổ {dqn_eval['window_open_pct']:.2f}% thay vì 100%. Đây là kết quả có ích để kiểm tra logic điều khiển, nhưng vẫn cần tinh chỉnh reward nếu mục tiêu là cân bằng tốt hơn giữa comfort và năng lượng."))
else:
    body.append(p("Chưa có file result.json của DQN tái hiện để chèn vào báo cáo."))

body.append(heading("6. Nhận xét sau khảo sát", 2))
body.append(p("Repo WHU-LX có giá trị làm nền tảng ý tưởng vì kết hợp mô hình dự báo chuyển trạng thái với DQN để tối ưu điều khiển. Cách thiết kế này phù hợp với HVAC vì action hiện tại có ảnh hưởng trễ đến nhiệt độ và comfort ở các timestep sau."))
body.append(p("Tuy nhiên, repo gốc chưa dùng EnergyPlus, Sinergym hoặc file thời tiết EPW. Dataset là occupant behavior CSV nên chưa mô phỏng trực tiếp tải HVAC theo khí hậu Việt Nam. Vì vậy, nếu muốn nghiên cứu điều khiển HVAC theo điều kiện HCM/Hà Nội, cần chuyển sang môi trường mô phỏng EnergyPlus/Sinergym và dùng weather file địa phương."))
body.append(p("State gốc phù hợp với HVAC + window control, nhưng chưa có occupancy trực tiếp và chưa có công suất HVAC làm input. Với hướng mô phỏng EnergyPlus, state nên được điều chỉnh để gồm thời gian, thời tiết, trạng thái trong phòng, occupancy và HVAC power."))

body.append(heading("7. Hướng đi tiếp theo", 2))
body.append(p("Hướng tiếp theo là kế thừa ý tưởng DQN từ repo WHU-LX, nhưng thay dataset occupant behavior bằng dữ liệu mô phỏng Sinergym/EnergyPlus. Trước mắt nên tập trung vào bài toán HVAC setpoint control, đặc biệt là cooling setpoint, thay vì điều khiển đồng thời cả cửa sổ."))
body.append(p("Pipeline đề xuất gồm: chọn weather file EPW địa phương, tạo dataset mô phỏng theo mùa, xác định state/action/reward, train DQN, evaluate full-season và so sánh với các baseline cố định hoặc rule-based. Sau khi pipeline ổn định, có thể mở rộng state/action space để gần hơn với bài toán gốc."))

body.append(heading("8. Phân chia công việc cho 3 thành viên", 2))
body.append(p("Công việc được chia thành 3 nhóm tương đối cân bằng: khảo sát lý thuyết/dataset, xây dựng pipeline mô phỏng, và huấn luyện - đánh giá mô hình."))
body.append(table(
    ["Thành viên", "Nhiệm vụ chính"],
    [
        ["Thành viên 1", "Khảo sát repo WHU-LX và tài liệu liên quan; tổng hợp keyword, bài toán HVAC/window control, dataset gốc, state/action/reward gốc; viết phần cơ sở lý thuyết và so sánh với hướng Sinergym/EnergyPlus."],
        ["Thành viên 2", "Chuẩn bị dữ liệu và môi trường mô phỏng; kiểm tra weather file EPW, Sinergym, EnergyPlus; chạy script generate data; thống kê dataset đầu vào, số dòng/cột, missing value, duplicate và các biến chính."],
        ["Thành viên 3", "Phụ trách DQN training và evaluation; kiểm tra train_rl.py, evaluate_performance.py; chạy baseline, phân tích energy saving, PMV/comfort violation; tổng hợp kết quả và đề xuất cải thiện reward/action space."],
    ],
))
body.append(p("Đầu ra cuối tuần gồm: báo cáo khảo sát repo gốc, mô tả dataset và hướng mô phỏng, bảng phân tích state/action/reward, kết quả tái hiện XGBoost + DQN một ngày mẫu và kế hoạch triển khai tuần tiếp theo."))

document_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body>"
    + "".join(body)
    + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" '
    'w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
    "</w:body></w:document>"
)

content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""

rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""

styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Arial" w:cs="Arial"/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="200" w:after="100"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="160" w:after="80"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="25"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListBullet">
    <w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr><w:tblBorders>
      <w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>
    </w:tblBorders></w:tblPr>
  </w:style>
</w:styles>
"""

with ZipFile(OUT, "w", ZIP_DEFLATED) as docx:
    docx.writestr("[Content_Types].xml", content_types)
    docx.writestr("_rels/.rels", rels)
    docx.writestr("word/_rels/document.xml.rels", document_rels)
    docx.writestr("word/document.xml", document_xml)
    docx.writestr("word/styles.xml", styles)

print(f"Wrote {OUT}")
