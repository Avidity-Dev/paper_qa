from re import I
from flask import Flask
from flask_cors import CORS


cors = CORS()


def create_app(config_class=DevelopmentConfig):
    app = Flask(__name__)
    app.config.from_object(config_class)

    cors.init_app(app, resources={r"/*": {"origins": "*"}})
    app.config["CORS_HEADERS"] = "Content-Type"

    """ Add Resources and Rules for API Endpoints """

    import api.src.views.patient as patient
    import api.src.views.physician as md
    import api.src.views.pharmacyorder as po
    import api.src.views.user as user
    import api.src.views.laborder as lo
    import api.src.views.equipment as eq

    equipment_view = eq.EquipmentAPI.as_view("equipment_api")
    app.add_url_rule(
        "/equipment/", defaults={"id": None}, view_func=equipment_view, methods=["GET"]
    )
    app.add_url_rule(
        "/equipment/",
        view_func=equipment_view,
        methods=[
            "POST",
        ],
    )
    app.add_url_rule("/equipment/<int:id>", view_func=equipment_view, methods=["GET"])

    laborder_view = lo.LabOrderAPI.as_view("laborder_api")
    app.add_url_rule(
        "/laborder/",
        defaults={"id": None},
        view_func=laborder_view,
        methods=["GET", "PUT", "DELETE"],
    )
    app.add_url_rule(
        "/laborder/",
        view_func=laborder_view,
        methods=[
            "POST",
        ],
    )
    app.add_url_rule(
        "/laborder/<int:id>", view_func=laborder_view, methods=["GET", "PUT", "DELETE"]
    )

    patient_view = patient.PatientAPI.as_view("patient_api")
    app.add_url_rule(
        "/records/", defaults={"id": None}, view_func=patient_view, methods=["GET"]
    )
    app.add_url_rule("/records/", view_func=patient_view, methods=["POST", "PUT"])
    app.add_url_rule("/records/<int:id>", view_func=patient_view, methods=["GET"])

    pharmacyorder_view = po.PharmacyOrderAPI.as_view("pharmacyorder_api")
    app.add_url_rule(
        "/pharmacy/order",
        defaults={"id": None},
        view_func=pharmacyorder_view,
        methods=["GET"],
    )
    app.add_url_rule(
        "/pharmacy/order", view_func=pharmacyorder_view, methods=["POST", "PUT"]
    )
    app.add_url_rule(
        "/pharmacy/order/<int:id>", view_func=pharmacyorder_view, methods=["GET"]
    )

    physician_view = md.PhysicianAPI.as_view("physician_api")
    app.add_url_rule(
        "/physicians/", defaults={"id": None}, view_func=physician_view, methods=["GET"]
    )
    app.add_url_rule(
        "/physicians/",
        view_func=physician_view,
        methods=[
            "POST",
        ],
    )
    app.add_url_rule("/physicians/<int:id>", view_func=physician_view, methods=["GET"])

    # routes relative to user creation, login, logout
    register_view = user.RegisterAPI.as_view("register_api")
    app.add_url_rule("/admin/registeruser", view_func=register_view, methods=["POST"])

    login_view = user.LoginAPI.as_view("login_api")
    app.add_url_rule("/login/", view_func=login_view, methods=["POST"])

    status_view = user.StatusAPI.as_view("status_api")
    app.add_url_rule("/auth/status/", view_func=status_view, methods=["GET"])

    logout_view = user.LogoutAPI.as_view("logout_api")
    app.add_url_rule("/dashboard/logout/", view_func=logout_view, methods=["POST"])

    userprofile_view = user.UserProfileAPI.as_view("userprofile_api")
    app.add_url_rule("/profile/", view_func=userprofile_view, methods=["PUT"])

    return app
