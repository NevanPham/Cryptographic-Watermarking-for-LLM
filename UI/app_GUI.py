# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.9.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QLabel,
    QMainWindow, QPlainTextEdit, QPushButton, QSizePolicy,
    QSlider, QStackedWidget, QTextBrowser, QWidget)
import resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1000, 600)
        MainWindow.setStyleSheet(u"background-color: rgb(250, 250, 250);\n"
"font-family: \"Inter\";\n"
"\n"
"QPushButton {\n"
"	border: none;\n"
"}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.navBar = QWidget(self.centralwidget)
        self.navBar.setObjectName(u"navBar")
        self.navBar.setGeometry(QRect(0, 0, 221, 601))
        self.navBar.setStyleSheet(u"background-color: rgb(250, 250, 250);\n"
"border-right: 1px solid rgb(210, 210, 210);")
        self.label = QLabel(self.navBar)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 17, 141, 21))
        font = QFont()
        font.setFamilies([u"Inter"])
        font.setPointSize(15)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setStyleSheet(u"border: none;")
        self.detectButton = QPushButton(self.navBar)
        self.detectButton.setObjectName(u"detectButton")
        self.detectButton.setGeometry(QRect(10, 90, 201, 31))
        font1 = QFont()
        font1.setFamilies([u"Inter"])
        font1.setPointSize(9)
        self.detectButton.setFont(font1)
        self.detectButton.setStyleSheet(u"QPushButton {\n"
"text-align: left;\n"
"padding-left: 15px;\n"
"font-family: \"Inter\";\n"
"border: none;\n"
"background-color: rgb(250,250,250);\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: lightgrey;\n"
"}\n"
"\n"
"QPushButton:checked {\n"
"background-color: rgb(255,218,167);\n"
"}")
        icon = QIcon()
        icon.addFile(u":/icons/files/glass.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.detectButton.setIcon(icon)
        self.detectButton.setCheckable(True)
        self.detectButton.setAutoExclusive(True)
        self.generateButton = QPushButton(self.navBar)
        self.generateButton.setObjectName(u"generateButton")
        self.generateButton.setGeometry(QRect(10, 50, 201, 31))
        self.generateButton.setFont(font1)
        self.generateButton.setStyleSheet(u"QPushButton {\n"
"text-align: left;\n"
"padding-left: 15px;\n"
"font-family: \"Inter\";\n"
"border: none;\n"
"background-color: rgb(250,250,250);\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: lightgrey;\n"
"}\n"
"\n"
"QPushButton:checked {\n"
"background-color: rgb(182,227,206);\n"
"}\n"
"")
        icon1 = QIcon()
        icon1.addFile(u":/icons/files/text.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.generateButton.setIcon(icon1)
        self.generateButton.setCheckable(True)
        self.generateButton.setAutoExclusive(True)
        self.evaluateButton = QPushButton(self.navBar)
        self.evaluateButton.setObjectName(u"evaluateButton")
        self.evaluateButton.setGeometry(QRect(10, 130, 201, 31))
        self.evaluateButton.setFont(font1)
        self.evaluateButton.setStyleSheet(u"QPushButton {\n"
"text-align: left;\n"
"padding-left: 15px;\n"
"font-family: \"Inter\";\n"
"border: none;\n"
"background-color: rgb(250,250,250);\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: lightgrey;\n"
"}\n"
"\n"
"QPushButton:checked {\n"
"background-color: rgb(199,236,255);\n"
"}")
        icon2 = QIcon()
        icon2.addFile(u":/icons/files/graph.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.evaluateButton.setIcon(icon2)
        self.evaluateButton.setCheckable(True)
        self.evaluateButton.setAutoExclusive(True)
        self.accountButton = QPushButton(self.navBar)
        self.accountButton.setObjectName(u"accountButton")
        self.accountButton.setGeometry(QRect(10, 560, 201, 31))
        self.accountButton.setFont(font1)
        self.accountButton.setStyleSheet(u"QPushButton {\n"
"text-align: left;\n"
"padding-left: 15px;\n"
"font-family: \"Inter\";\n"
"border: 1px solid lightgrey;\n"
"background-color: rgb(250,250,250);\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: lightgrey;\n"
"}\n"
"")
        icon3 = QIcon(QIcon.fromTheme(u"emblem-shared"))
        self.accountButton.setIcon(icon3)
        self.accountButton.setCheckable(True)
        self.accountButton.setAutoExclusive(False)
        self.accountWidget = QWidget(self.navBar)
        self.accountWidget.setObjectName(u"accountWidget")
        self.accountWidget.setGeometry(QRect(10, 370, 201, 191))
        self.accountWidget.setStyleSheet(u"border: 1px solid lightgrey;\n"
"border-radius: 4px;")
        self.label_27 = QLabel(self.accountWidget)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setGeometry(QRect(20, 10, 131, 20))
        self.label_27.setStyleSheet(u"border: none;")
        self.firstNameBox = QPlainTextEdit(self.accountWidget)
        self.firstNameBox.setObjectName(u"firstNameBox")
        self.firstNameBox.setGeometry(QRect(20, 30, 161, 30))
        self.firstNameBox.setStyleSheet(u"background-color: white;\n"
"border: 1px solid lightgrey;\n"
"border-radius: 6px 6px;\n"
"color: black;\n"
"padding-top: 4px;\n"
"padding-left: 2px;")
        self.saveDetailsButton = QPushButton(self.accountWidget)
        self.saveDetailsButton.setObjectName(u"saveDetailsButton")
        self.saveDetailsButton.setGeometry(QRect(20, 120, 91, 31))
        self.saveDetailsButton.setStyleSheet(u"QPushButton {\n"
"text-align: center;\n"
"font-family: \"Inter\";\n"
"border: none;\n"
"background-color: lightgrey;\n"
"color: black;\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: silver;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: grey;\n"
"}\n"
"")
        self.accounDetailsMessage = QLabel(self.accountWidget)
        self.accounDetailsMessage.setObjectName(u"accounDetailsMessage")
        self.accounDetailsMessage.setGeometry(QRect(20, 160, 121, 20))
        self.accounDetailsMessage.setStyleSheet(u"border: none;")
        self.label_28 = QLabel(self.accountWidget)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setGeometry(QRect(20, 60, 131, 20))
        self.label_28.setStyleSheet(u"border: none;")
        self.lastNameBox = QPlainTextEdit(self.accountWidget)
        self.lastNameBox.setObjectName(u"lastNameBox")
        self.lastNameBox.setGeometry(QRect(20, 80, 161, 30))
        self.lastNameBox.setStyleSheet(u"background-color: white;\n"
"border: 1px solid lightgrey;\n"
"border-radius: 6px 6px;\n"
"color: black;\n"
"padding-top: 4px;\n"
"padding-left: 2px;")
        self.loginWidget = QWidget(self.navBar)
        self.loginWidget.setObjectName(u"loginWidget")
        self.loginWidget.setGeometry(QRect(10, 370, 201, 191))
        self.loginWidget.setStyleSheet(u"border: 1px solid lightgrey;\n"
"border-radius: 4px;")
        self.label_26 = QLabel(self.loginWidget)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setGeometry(QRect(20, 60, 131, 20))
        self.label_26.setStyleSheet(u"border: none;")
        self.label_21 = QLabel(self.loginWidget)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(20, 10, 121, 20))
        self.label_21.setStyleSheet(u"border: none;")
        self.usernameBox = QPlainTextEdit(self.loginWidget)
        self.usernameBox.setObjectName(u"usernameBox")
        self.usernameBox.setGeometry(QRect(20, 30, 161, 30))
        self.usernameBox.setStyleSheet(u"background-color: white;\n"
"border: 1px solid lightgrey;\n"
"border-radius: 6px 6px;\n"
"color: black;\n"
"padding-top: 4px;\n"
"padding-left: 2px;")
        self.passwordBox = QPlainTextEdit(self.loginWidget)
        self.passwordBox.setObjectName(u"passwordBox")
        self.passwordBox.setGeometry(QRect(20, 80, 161, 30))
        self.passwordBox.setStyleSheet(u"background-color: white;\n"
"border: 1px solid lightgrey;\n"
"border-radius: 6px 6px;\n"
"color: black;\n"
"padding-top: 4px;\n"
"padding-left: 2px;")
        self.loginButton = QPushButton(self.loginWidget)
        self.loginButton.setObjectName(u"loginButton")
        self.loginButton.setGeometry(QRect(20, 120, 91, 31))
        self.loginButton.setStyleSheet(u"QPushButton {\n"
"text-align: center;\n"
"font-family: \"Inter\";\n"
"border: none;\n"
"background-color: lightgrey;\n"
"color: black;\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: silver;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: grey;\n"
"}\n"
"")
        self.loginMessage = QLabel(self.loginWidget)
        self.loginMessage.setObjectName(u"loginMessage")
        self.loginMessage.setGeometry(QRect(20, 160, 121, 20))
        self.loginMessage.setStyleSheet(u"border: none;")
        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setGeometry(QRect(229, -1, 761, 601))
        self.stackedWidget.setStyleSheet(u"background-color: white;")
        self.generatePage = QWidget()
        self.generatePage.setObjectName(u"generatePage")
        self.generatePage.setStyleSheet(u"background-color: rgb(250,250,250);\n"
"")
        self.title = QLabel(self.generatePage)
        self.title.setObjectName(u"title")
        self.title.setGeometry(QRect(40, 20, 281, 41))
        font2 = QFont()
        font2.setFamilies([u"Inter"])
        font2.setPointSize(24)
        font2.setBold(True)
        self.title.setFont(font2)
        self.title.setStyleSheet(u"color: rgb(0,153,81);")
        self.title2 = QLabel(self.generatePage)
        self.title2.setObjectName(u"title2")
        self.title2.setGeometry(QRect(195, 20, 281, 41))
        self.title2.setFont(font2)
        self.promptBox = QPlainTextEdit(self.generatePage)
        self.promptBox.setObjectName(u"promptBox")
        self.promptBox.setGeometry(QRect(40, 80, 571, 31))
        self.promptBox.setStyleSheet(u"background-color: white;\n"
"border: 1px solid lightgrey;\n"
"border-radius: 6px 6px;\n"
"color: black;\n"
"padding-top: 4px;\n"
"padding-left: 2px;")
        self.label_1 = QLabel(self.generatePage)
        self.label_1.setObjectName(u"label_1")
        self.label_1.setGeometry(QRect(40, 130, 101, 20))
        self.genModelComboBox = QComboBox(self.generatePage)
        self.genModelComboBox.setObjectName(u"genModelComboBox")
        self.genModelComboBox.setGeometry(QRect(40, 150, 161, 22))
        self.genModelComboBox.setStyleSheet(u"QComboBox {\n"
"    background: white;\n"
"    border: 1px solid lightgrey;\n"
"    border-radius: 4px;\n"
"    padding: 3px;\n"
"padding-left: 5px;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    border: none;\n"
"    width: 20px;\n"
"    background: white;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	image: url(:/icons/files/arrow.png);\n"
"width: 12px;\n"
"}")
        self.genModelComboBox.setFrame(True)
        self.label_2 = QLabel(self.generatePage)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(40, 190, 131, 20))
        self.responseBox = QTextBrowser(self.generatePage)
        self.responseBox.setObjectName(u"responseBox")
        self.responseBox.setGeometry(QRect(40, 210, 681, 361))
        self.responseBox.setStyleSheet(u"background-color: white;\n"
"border: 1px solid rgb(0,153,81);\n"
"border-radius: 6px 6px;\n"
"padding-top: 4px;\n"
"padding-left: 2px;")
        self.genButton = QPushButton(self.generatePage)
        self.genButton.setObjectName(u"genButton")
        self.genButton.setGeometry(QRect(630, 80, 91, 31))
        self.genButton.setStyleSheet(u"QPushButton {\n"
"text-align: left;\n"
"font-family: \"Inter\";\n"
"border: 1px solid lightgrey;\n"
"background-color: white;\n"
"color: grey;\n"
"border-radius: 7px 7px;\n"
"padding-left: 8px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: whitesmoke;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: lightgrey;\n"
"}")
        icon4 = QIcon()
        icon4.addFile(u":/icons/files/send.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.genButton.setIcon(icon4)
        self.genAdvancedWidget = QWidget(self.generatePage)
        self.genAdvancedWidget.setObjectName(u"genAdvancedWidget")
        self.genAdvancedWidget.setGeometry(QRect(240, 170, 391, 251))
        self.genAdvancedWidget.setStyleSheet(u"border: 1px solid lightgrey;\n"
"border-radius: 4px;")
        self.generateUploadFileButton = QPushButton(self.genAdvancedWidget)
        self.generateUploadFileButton.setObjectName(u"generateUploadFileButton")
        self.generateUploadFileButton.setGeometry(QRect(20, 70, 161, 21))
        self.generateUploadFileButton.setStyleSheet(u"QPushButton {\n"
"text-align: left;\n"
"font-family: \"Inter\";\n"
"border: 1px solid lightgrey;\n"
"background-color: white;\n"
"color: black;\n"
"border-radius: 4px;\n"
"padding-left: 8px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: whitesmoke;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: lightgrey;\n"
"}")
        icon5 = QIcon()
        icon5.addFile(u":/icons/files/folder.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.generateUploadFileButton.setIcon(icon5)
        self.genMaxNewTokensSlider = QSlider(self.genAdvancedWidget)
        self.genMaxNewTokensSlider.setObjectName(u"genMaxNewTokensSlider")
        self.genMaxNewTokensSlider.setGeometry(QRect(210, 120, 160, 22))
        self.genMaxNewTokensSlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"QSlider {\n"
"border: none;\n"
"}")
        self.genMaxNewTokensSlider.setMinimum(512)
        self.genMaxNewTokensSlider.setMaximum(4096)
        self.genMaxNewTokensSlider.setOrientation(Qt.Orientation.Horizontal)
        self.label_13 = QLabel(self.genAdvancedWidget)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(210, 100, 141, 20))
        self.label_13.setStyleSheet(u"border: none;")
        self.label_16 = QLabel(self.genAdvancedWidget)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(20, 50, 121, 20))
        self.label_16.setStyleSheet(u"border: none;")
        self.genWaterMarkCheckBox = QCheckBox(self.genAdvancedWidget)
        self.genWaterMarkCheckBox.setObjectName(u"genWaterMarkCheckBox")
        self.genWaterMarkCheckBox.setGeometry(QRect(20, 20, 131, 20))
        self.genWaterMarkCheckBox.setStyleSheet(u"border: none;")
        self.genHashingSlider = QSlider(self.genAdvancedWidget)
        self.genHashingSlider.setObjectName(u"genHashingSlider")
        self.genHashingSlider.setGeometry(QRect(20, 120, 160, 22))
        self.genHashingSlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider {\n"
"border: none;\n"
"}")
        self.genHashingSlider.setMinimum(1)
        self.genHashingSlider.setMaximum(10)
        self.genHashingSlider.setOrientation(Qt.Orientation.Horizontal)
        self.label_17 = QLabel(self.genAdvancedWidget)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(20, 100, 131, 20))
        self.label_17.setStyleSheet(u"border: none;")
        self.genDeltaSlider = QSlider(self.genAdvancedWidget)
        self.genDeltaSlider.setObjectName(u"genDeltaSlider")
        self.genDeltaSlider.setGeometry(QRect(20, 190, 160, 22))
        self.genDeltaSlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"QSlider {\n"
"border: none;\n"
"}")
        self.genDeltaSlider.setMinimum(1)
        self.genDeltaSlider.setMaximum(5)
        self.genDeltaSlider.setSingleStep(1)
        self.genDeltaSlider.setPageStep(1)
        self.genDeltaSlider.setOrientation(Qt.Orientation.Horizontal)
        self.label_18 = QLabel(self.genAdvancedWidget)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(20, 170, 41, 20))
        self.label_18.setStyleSheet(u"border: none;")
        self.genEntropySlider = QSlider(self.genAdvancedWidget)
        self.genEntropySlider.setObjectName(u"genEntropySlider")
        self.genEntropySlider.setGeometry(QRect(210, 190, 160, 22))
        self.genEntropySlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"QSlider {\n"
"border: none;\n"
"}")
        self.genEntropySlider.setMinimum(1)
        self.genEntropySlider.setMaximum(6)
        self.genEntropySlider.setOrientation(Qt.Orientation.Horizontal)
        self.label_19 = QLabel(self.genAdvancedWidget)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(210, 170, 131, 20))
        self.label_19.setStyleSheet(u"border: none;")
        self.genHashingDisplayValue = QLabel(self.genAdvancedWidget)
        self.genHashingDisplayValue.setObjectName(u"genHashingDisplayValue")
        self.genHashingDisplayValue.setGeometry(QRect(20, 150, 31, 16))
        self.genHashingDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.genDeltaDisplayValue = QLabel(self.genAdvancedWidget)
        self.genDeltaDisplayValue.setObjectName(u"genDeltaDisplayValue")
        self.genDeltaDisplayValue.setGeometry(QRect(20, 220, 31, 16))
        self.genDeltaDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.genMaxDisplayValue = QLabel(self.genAdvancedWidget)
        self.genMaxDisplayValue.setObjectName(u"genMaxDisplayValue")
        self.genMaxDisplayValue.setGeometry(QRect(210, 150, 31, 16))
        self.genMaxDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.genEntropyDisplayValue = QLabel(self.genAdvancedWidget)
        self.genEntropyDisplayValue.setObjectName(u"genEntropyDisplayValue")
        self.genEntropyDisplayValue.setGeometry(QRect(210, 220, 31, 16))
        self.genEntropyDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.genAdvancedButton = QPushButton(self.generatePage)
        self.genAdvancedButton.setObjectName(u"genAdvancedButton")
        self.genAdvancedButton.setGeometry(QRect(240, 140, 131, 31))
        self.genAdvancedButton.setStyleSheet(u"QPushButton {\n"
"text-align: center;\n"
"font-family: \"Inter\";\n"
"border: none;\n"
"background-color: rgb(182,227,206);\n"
"color: black;\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: #6fd9a7;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color :#57c994;\n"
"}\n"
"")
        self.genAdvancedButton.setCheckable(True)
        self.stackedWidget.addWidget(self.generatePage)
        self.detectPage = QWidget()
        self.detectPage.setObjectName(u"detectPage")
        self.detectPage.setStyleSheet(u"background-color: rgb(250,250,250);")
        self.title2_3 = QLabel(self.detectPage)
        self.title2_3.setObjectName(u"title2_3")
        self.title2_3.setGeometry(QRect(155, 20, 281, 41))
        self.title2_3.setFont(font2)
        self.title_3 = QLabel(self.detectPage)
        self.title_3.setObjectName(u"title_3")
        self.title_3.setGeometry(QRect(40, 20, 281, 41))
        self.title_3.setFont(font2)
        self.title_3.setStyleSheet(u"color: #DD8F22;")
        self.label_11 = QLabel(self.detectPage)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(40, 70, 131, 20))
        self.checkButton = QPushButton(self.detectPage)
        self.checkButton.setObjectName(u"checkButton")
        self.checkButton.setGeometry(QRect(40, 350, 91, 31))
        self.checkButton.setStyleSheet(u"QPushButton {\n"
"text-align: center;\n"
"font-family: \"Inter\";\n"
"border: none;\n"
"background-color: #ffd59c;\n"
"color: black;\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: #ffcc87;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: #ffc370;\n"
"}")
        self.label_12 = QLabel(self.detectPage)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(40, 500, 141, 21))
        self.label_12.setFont(font)
        self.label_12.setStyleSheet(u"border: none;")
        self.resultsDisplay = QLabel(self.detectPage)
        self.resultsDisplay.setObjectName(u"resultsDisplay")
        self.resultsDisplay.setGeometry(QRect(40, 520, 251, 61))
        font3 = QFont()
        font3.setFamilies([u"Inter"])
        font3.setPointSize(12)
        font3.setBold(False)
        self.resultsDisplay.setFont(font3)
        self.resultsDisplay.setStyleSheet(u"border: none;")
        self.userDisplay = QTextBrowser(self.detectPage)
        self.userDisplay.setObjectName(u"userDisplay")
        self.userDisplay.setGeometry(QRect(270, 540, 201, 25))
        self.userDisplay.setStyleSheet(u"text-align: left;\n"
"font-family: \"Inter\";\n"
"border: 1px solid lightgrey;\n"
"background-color: white;\n"
"color: black;\n"
"border-radius: 4px;\n"
"padding-left: 8px;")
        self.label_15 = QLabel(self.detectPage)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(270, 520, 131, 20))
        self.detectBox = QPlainTextEdit(self.detectPage)
        self.detectBox.setObjectName(u"detectBox")
        self.detectBox.setGeometry(QRect(40, 90, 681, 251))
        self.detectBox.setStyleSheet(u"background-color: white;\n"
"border: 1px solid #DD8F22;\n"
"border-radius: 6px 6px;\n"
"padding-top: 4px;\n"
"padding-left: 2px;")
        self.detectAdvancedButton = QPushButton(self.detectPage)
        self.detectAdvancedButton.setObjectName(u"detectAdvancedButton")
        self.detectAdvancedButton.setGeometry(QRect(250, 350, 131, 31))
        self.detectAdvancedButton.setStyleSheet(u"QPushButton {\n"
"text-align: center;\n"
"font-family: \"Inter\";\n"
"border: none;\n"
"background-color: #ffd59c;\n"
"color: black;\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: #ffcc87;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: #ffc370;\n"
"}\n"
"")
        self.detectAdvancedButton.setCheckable(True)
        self.detectAdvancedWidget = QWidget(self.detectPage)
        self.detectAdvancedWidget.setObjectName(u"detectAdvancedWidget")
        self.detectAdvancedWidget.setGeometry(QRect(250, 380, 391, 201))
        self.detectAdvancedWidget.setStyleSheet(u"border: 1px solid lightgrey;\n"
"border-radius: 4px;")
        self.detectHashingSlider = QSlider(self.detectAdvancedWidget)
        self.detectHashingSlider.setObjectName(u"detectHashingSlider")
        self.detectHashingSlider.setGeometry(QRect(20, 80, 160, 22))
        self.detectHashingSlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"QSlider {\n"
"border: none;\n"
"}")
        self.detectHashingSlider.setOrientation(Qt.Orientation.Horizontal)
        self.label_23 = QLabel(self.detectAdvancedWidget)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(20, 60, 131, 20))
        self.label_23.setStyleSheet(u"border: none;")
        self.detectZSlider = QSlider(self.detectAdvancedWidget)
        self.detectZSlider.setObjectName(u"detectZSlider")
        self.detectZSlider.setGeometry(QRect(20, 150, 160, 22))
        self.detectZSlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"QSlider {\n"
"border: none;\n"
"}")
        self.detectZSlider.setOrientation(Qt.Orientation.Horizontal)
        self.label_24 = QLabel(self.detectAdvancedWidget)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setGeometry(QRect(20, 130, 131, 20))
        self.label_24.setStyleSheet(u"border: none;")
        self.detectEntropySlider = QSlider(self.detectAdvancedWidget)
        self.detectEntropySlider.setObjectName(u"detectEntropySlider")
        self.detectEntropySlider.setGeometry(QRect(210, 80, 160, 22))
        self.detectEntropySlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"QSlider {\n"
"border: none;\n"
"}")
        self.detectEntropySlider.setOrientation(Qt.Orientation.Horizontal)
        self.label_25 = QLabel(self.detectAdvancedWidget)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setGeometry(QRect(210, 60, 131, 20))
        self.label_25.setStyleSheet(u"border: none;")
        self.detectUploadFileButton = QPushButton(self.detectAdvancedWidget)
        self.detectUploadFileButton.setObjectName(u"detectUploadFileButton")
        self.detectUploadFileButton.setGeometry(QRect(20, 30, 161, 21))
        self.detectUploadFileButton.setStyleSheet(u"QPushButton {\n"
"text-align: left;\n"
"font-family: \"Inter\";\n"
"border: 1px solid lightgrey;\n"
"background-color: white;\n"
"color: black;\n"
"border-radius: 4px;\n"
"padding-left: 8px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: whitesmoke;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: lightgrey;\n"
"}")
        self.detectUploadFileButton.setIcon(icon5)
        self.label_20 = QLabel(self.detectAdvancedWidget)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(20, 10, 121, 20))
        self.label_20.setStyleSheet(u"border: none;")
        self.detectHashingDisplayValue = QLabel(self.detectAdvancedWidget)
        self.detectHashingDisplayValue.setObjectName(u"detectHashingDisplayValue")
        self.detectHashingDisplayValue.setGeometry(QRect(20, 110, 31, 16))
        self.detectHashingDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.detectEntropyDisplayValue = QLabel(self.detectAdvancedWidget)
        self.detectEntropyDisplayValue.setObjectName(u"detectEntropyDisplayValue")
        self.detectEntropyDisplayValue.setGeometry(QRect(210, 110, 31, 16))
        self.detectEntropyDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.detectZDisplayValue = QLabel(self.detectAdvancedWidget)
        self.detectZDisplayValue.setObjectName(u"detectZDisplayValue")
        self.detectZDisplayValue.setGeometry(QRect(20, 180, 31, 16))
        self.detectZDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.label_14 = QLabel(self.detectAdvancedWidget)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(210, 10, 101, 20))
        self.label_14.setStyleSheet(u"border: none;")
        self.detectModelComboBox = QComboBox(self.detectAdvancedWidget)
        self.detectModelComboBox.setObjectName(u"detectModelComboBox")
        self.detectModelComboBox.setGeometry(QRect(210, 30, 161, 22))
        self.detectModelComboBox.setStyleSheet(u"QComboBox {\n"
"    background: white;\n"
"    border: 1px solid lightgrey;\n"
"    border-radius: 4px;\n"
"    padding: 3px;\n"
"padding-left: 5px;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    border: none;\n"
"    width: 20px;\n"
"    background: white;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	image: url(:/icons/files/arrow.png);\n"
"width: 12px;\n"
"}")
        self.detectModelComboBox.setFrame(True)
        self.stackedWidget.addWidget(self.detectPage)
        self.title_3.raise_()
        self.title2_3.raise_()
        self.label_11.raise_()
        self.checkButton.raise_()
        self.label_12.raise_()
        self.resultsDisplay.raise_()
        self.userDisplay.raise_()
        self.label_15.raise_()
        self.detectBox.raise_()
        self.detectAdvancedButton.raise_()
        self.detectAdvancedWidget.raise_()
        self.evaluatePage = QWidget()
        self.evaluatePage.setObjectName(u"evaluatePage")
        self.evaluatePage.setStyleSheet(u"background-color: rgb(250,250,250);\n"
"")
        self.title2_2 = QLabel(self.evaluatePage)
        self.title2_2.setObjectName(u"title2_2")
        self.title2_2.setGeometry(QRect(188, 20, 281, 41))
        self.title2_2.setFont(font2)
        self.title_2 = QLabel(self.evaluatePage)
        self.title_2.setObjectName(u"title_2")
        self.title_2.setGeometry(QRect(40, 20, 281, 41))
        self.title_2.setFont(font2)
        self.title_2.setStyleSheet(u"color: rgb(91,177,219);")
        self.label_3 = QLabel(self.evaluatePage)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(240, 80, 101, 20))
        self.evalModelComboBox = QComboBox(self.evaluatePage)
        self.evalModelComboBox.setObjectName(u"evalModelComboBox")
        self.evalModelComboBox.setGeometry(QRect(240, 100, 161, 22))
        self.evalModelComboBox.setStyleSheet(u"QComboBox {\n"
"    background: white;\n"
"    border: 1px solid lightgrey;\n"
"    border-radius: 4px;\n"
"    padding: 3px;\n"
"padding-left: 5px;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    border: none;\n"
"    width: 20px;\n"
"    background: white;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	image: url(:/icons/files/arrow.png);\n"
"width: 12px;\n"
"}")
        self.evalModelComboBox.setFrame(True)
        self.evalUploadFileButton = QPushButton(self.evaluatePage)
        self.evalUploadFileButton.setObjectName(u"evalUploadFileButton")
        self.evalUploadFileButton.setGeometry(QRect(40, 100, 161, 21))
        self.evalUploadFileButton.setStyleSheet(u"QPushButton {\n"
"text-align: left;\n"
"font-family: \"Inter\";\n"
"border: 1px solid lightgrey;\n"
"background-color: white;\n"
"color: black;\n"
"border-radius: 4px;\n"
"padding-left: 8px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: whitesmoke;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: lightgrey;\n"
"}")
        self.evalUploadFileButton.setIcon(icon5)
        self.label_4 = QLabel(self.evaluatePage)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(40, 80, 121, 20))
        self.evalButton = QPushButton(self.evaluatePage)
        self.evalButton.setObjectName(u"evalButton")
        self.evalButton.setGeometry(QRect(440, 90, 91, 31))
        self.evalButton.setStyleSheet(u"QPushButton {\n"
"text-align: center;\n"
"font-family: \"Inter\";\n"
"border: none;\n"
"background-color: rgb(199,236,255);\n"
"color: black;\n"
"border-radius: 7px 7px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: #a1dfff;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: #78d1ff;\n"
"}\n"
"")
        self.label_5 = QLabel(self.evaluatePage)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(40, 190, 141, 21))
        self.label_5.setFont(font)
        self.label_5.setStyleSheet(u"border: none;")
        self.evalMaxNewTokensSlider = QSlider(self.evaluatePage)
        self.evalMaxNewTokensSlider.setObjectName(u"evalMaxNewTokensSlider")
        self.evalMaxNewTokensSlider.setGeometry(QRect(240, 310, 160, 22))
        self.evalMaxNewTokensSlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"")
        self.evalMaxNewTokensSlider.setMinimum(512)
        self.evalMaxNewTokensSlider.setMaximum(4096)
        self.evalMaxNewTokensSlider.setOrientation(Qt.Orientation.Horizontal)
        self.label_6 = QLabel(self.evaluatePage)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(40, 220, 101, 20))
        self.label_7 = QLabel(self.evaluatePage)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(40, 290, 101, 20))
        self.label_8 = QLabel(self.evaluatePage)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(240, 290, 131, 20))
        self.label_9 = QLabel(self.evaluatePage)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(440, 220, 101, 20))
        self.graph1 = QWidget(self.evaluatePage)
        self.graph1.setObjectName(u"graph1")
        self.graph1.setGeometry(QRect(40, 370, 330, 200))
        self.graph1.setStyleSheet(u"border: 1px solid lightgrey;\n"
"border-radius: 5px 5px;")
        self.graph2 = QWidget(self.evaluatePage)
        self.graph2.setObjectName(u"graph2")
        self.graph2.setGeometry(QRect(400, 370, 330, 200))
        self.graph2.setStyleSheet(u"border: 1px solid lightgrey;\n"
"border-radius: 5px 5px;")
        self.evalMaxDisplayValue = QLabel(self.evaluatePage)
        self.evalMaxDisplayValue.setObjectName(u"evalMaxDisplayValue")
        self.evalMaxDisplayValue.setGeometry(QRect(240, 340, 31, 16))
        self.evalMaxDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.evalZDisplayValue = QLabel(self.evaluatePage)
        self.evalZDisplayValue.setObjectName(u"evalZDisplayValue")
        self.evalZDisplayValue.setGeometry(QRect(240, 270, 31, 16))
        self.evalZDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.label_22 = QLabel(self.evaluatePage)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(240, 220, 101, 20))
        self.evalZSlider = QSlider(self.evaluatePage)
        self.evalZSlider.setObjectName(u"evalZSlider")
        self.evalZSlider.setGeometry(QRect(240, 240, 160, 22))
        self.evalZSlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"")
        self.evalZSlider.setOrientation(Qt.Orientation.Horizontal)
        self.evalDeltaDisplayValue = QLabel(self.evaluatePage)
        self.evalDeltaDisplayValue.setObjectName(u"evalDeltaDisplayValue")
        self.evalDeltaDisplayValue.setGeometry(QRect(40, 270, 31, 16))
        self.evalDeltaDisplayValue.setStyleSheet(u"border: none;\n"
"color: grey;")
        self.evalDeltaSlider = QSlider(self.evaluatePage)
        self.evalDeltaSlider.setObjectName(u"evalDeltaSlider")
        self.evalDeltaSlider.setGeometry(QRect(40, 240, 160, 22))
        self.evalDeltaSlider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"    border: none;\n"
"    background: #ccc;            \n"
"    height: 3px;                 \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: black;         \n"
"    border: none;\n"
"    width: 12px;                \n"
"    height: 12px;\n"
"    margin: -5px 0;              \n"
"    border-radius: 6px;          \n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #78C4EA;         \n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #3498db;        \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #e0e0e0;         \n"
"    border-radius: 1.5px;\n"
"}\n"
"\n"
"")
        self.evalDeltaSlider.setMaximum(5)
        self.evalDeltaSlider.setOrientation(Qt.Orientation.Horizontal)
        self.label_29 = QLabel(self.evaluatePage)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setGeometry(QRect(440, 290, 101, 20))
        self.evalSweepComboBox = QComboBox(self.evaluatePage)
        self.evalSweepComboBox.addItem("")
        self.evalSweepComboBox.addItem("")
        self.evalSweepComboBox.addItem("")
        self.evalSweepComboBox.setObjectName(u"evalSweepComboBox")
        self.evalSweepComboBox.setGeometry(QRect(440, 310, 161, 22))
        self.evalSweepComboBox.setStyleSheet(u"QComboBox {\n"
"    background: white;\n"
"    border: 1px solid lightgrey;\n"
"    border-radius: 4px;\n"
"    padding: 3px;\n"
"padding-left: 5px;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    border: none;\n"
"    width: 20px;\n"
"    background: white;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	image: url(:/icons/files/arrow.png);\n"
"width: 12px;\n"
"}")
        self.evalSweepComboBox.setFrame(True)
        self.evalDownloadFileButton = QPushButton(self.evaluatePage)
        self.evalDownloadFileButton.setObjectName(u"evalDownloadFileButton")
        self.evalDownloadFileButton.setGeometry(QRect(40, 150, 161, 21))
        self.evalDownloadFileButton.setStyleSheet(u"QPushButton {\n"
"text-align: left;\n"
"font-family: \"Inter\";\n"
"border: 1px solid lightgrey;\n"
"background-color: white;\n"
"color: black;\n"
"border-radius: 4px;\n"
"padding-left: 8px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"background-color: whitesmoke;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: lightgrey;\n"
"}")
        self.evalDownloadFileButton.setIcon(icon5)
        self.label_10 = QLabel(self.evaluatePage)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(40, 130, 121, 20))
        self.entWidget = QWidget(self.evaluatePage)
        self.entWidget.setObjectName(u"entWidget")
        self.entWidget.setGeometry(QRect(440, 240, 151, 41))
        self.evalEnt1Check = QCheckBox(self.entWidget)
        self.evalEnt1Check.setObjectName(u"evalEnt1Check")
        self.evalEnt1Check.setGeometry(QRect(0, 0, 81, 20))
        self.evalEnt2Check = QCheckBox(self.entWidget)
        self.evalEnt2Check.setObjectName(u"evalEnt2Check")
        self.evalEnt2Check.setGeometry(QRect(50, 0, 81, 20))
        self.evalEnt3Check = QCheckBox(self.entWidget)
        self.evalEnt3Check.setObjectName(u"evalEnt3Check")
        self.evalEnt3Check.setGeometry(QRect(100, 0, 51, 20))
        self.evalEnt4Check = QCheckBox(self.entWidget)
        self.evalEnt4Check.setObjectName(u"evalEnt4Check")
        self.evalEnt4Check.setGeometry(QRect(0, 20, 81, 20))
        self.evalEnt5Check = QCheckBox(self.entWidget)
        self.evalEnt5Check.setObjectName(u"evalEnt5Check")
        self.evalEnt5Check.setGeometry(QRect(50, 20, 81, 20))
        self.evalEnt6Check = QCheckBox(self.entWidget)
        self.evalEnt6Check.setObjectName(u"evalEnt6Check")
        self.evalEnt6Check.setGeometry(QRect(100, 20, 81, 20))
        self.hashWidget = QWidget(self.evaluatePage)
        self.hashWidget.setObjectName(u"hashWidget")
        self.hashWidget.setGeometry(QRect(30, 310, 161, 51))
        self.evalHash8Check = QCheckBox(self.hashWidget)
        self.evalHash8Check.setObjectName(u"evalHash8Check")
        self.evalHash8Check.setGeometry(QRect(110, 20, 81, 20))
        self.evalHash6Check = QCheckBox(self.hashWidget)
        self.evalHash6Check.setObjectName(u"evalHash6Check")
        self.evalHash6Check.setGeometry(QRect(10, 20, 81, 20))
        self.evalHash7Check = QCheckBox(self.hashWidget)
        self.evalHash7Check.setObjectName(u"evalHash7Check")
        self.evalHash7Check.setGeometry(QRect(60, 20, 81, 20))
        self.evalHash5Check = QCheckBox(self.hashWidget)
        self.evalHash5Check.setObjectName(u"evalHash5Check")
        self.evalHash5Check.setGeometry(QRect(110, 0, 81, 20))
        self.evalHash4Check = QCheckBox(self.hashWidget)
        self.evalHash4Check.setObjectName(u"evalHash4Check")
        self.evalHash4Check.setGeometry(QRect(60, 0, 81, 20))
        self.evalHash3Check = QCheckBox(self.hashWidget)
        self.evalHash3Check.setObjectName(u"evalHash3Check")
        self.evalHash3Check.setGeometry(QRect(10, 0, 81, 20))
        self.evalHash6Check.raise_()
        self.evalHash7Check.raise_()
        self.evalHash3Check.raise_()
        self.evalHash4Check.raise_()
        self.evalHash5Check.raise_()
        self.evalHash8Check.raise_()
        self.stackedWidget.addWidget(self.evaluatePage)
        self.title_2.raise_()
        self.title2_2.raise_()
        self.label_3.raise_()
        self.evalModelComboBox.raise_()
        self.evalUploadFileButton.raise_()
        self.label_4.raise_()
        self.evalButton.raise_()
        self.label_5.raise_()
        self.evalMaxNewTokensSlider.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        self.graph1.raise_()
        self.graph2.raise_()
        self.evalMaxDisplayValue.raise_()
        self.evalZDisplayValue.raise_()
        self.label_22.raise_()
        self.evalZSlider.raise_()
        self.evalDeltaDisplayValue.raise_()
        self.evalDeltaSlider.raise_()
        self.label_29.raise_()
        self.evalSweepComboBox.raise_()
        self.evalDownloadFileButton.raise_()
        self.label_10.raise_()
        self.entWidget.raise_()
        self.hashWidget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"AI GenTools", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"AI GenTools", None))
        self.detectButton.setText(QCoreApplication.translate("MainWindow", u"  Detect", None))
        self.generateButton.setText(QCoreApplication.translate("MainWindow", u"  Generate", None))
        self.evaluateButton.setText(QCoreApplication.translate("MainWindow", u"  Evaluate", None))
        self.accountButton.setText(QCoreApplication.translate("MainWindow", u"  Account", None))
        self.label_27.setText(QCoreApplication.translate("MainWindow", u"First Name", None))
        self.firstNameBox.setPlainText("")
        self.saveDetailsButton.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.accounDetailsMessage.setText(QCoreApplication.translate("MainWindow", u"! Message", None))
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"Last Name", None))
        self.lastNameBox.setPlainText("")
        self.label_26.setText(QCoreApplication.translate("MainWindow", u"Password", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"Username", None))
        self.usernameBox.setPlainText("")
        self.passwordBox.setPlainText("")
        self.loginButton.setText(QCoreApplication.translate("MainWindow", u"Login", None))
        self.loginMessage.setText(QCoreApplication.translate("MainWindow", u"! Message", None))
        self.title.setText(QCoreApplication.translate("MainWindow", u"Generate", None))
        self.title2.setText(QCoreApplication.translate("MainWindow", u"content", None))
        self.promptBox.setPlainText("")
        self.label_1.setText(QCoreApplication.translate("MainWindow", u" Choose a model:", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u" Generated response:", None))
        self.genButton.setText(QCoreApplication.translate("MainWindow", u"Generate", None))
        self.generateUploadFileButton.setText(QCoreApplication.translate("MainWindow", u"  Upload...", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Maximum New Tokens", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u" Key file location:", None))
        self.genWaterMarkCheckBox.setText(QCoreApplication.translate("MainWindow", u"Watermark", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Hashing Context", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Delta", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Entropy Threshold", None))
        self.genHashingDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.genDeltaDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.genMaxDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.genEntropyDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.genAdvancedButton.setText(QCoreApplication.translate("MainWindow", u"Advanced Options", None))
        self.title2_3.setText(QCoreApplication.translate("MainWindow", u"watermark", None))
        self.title_3.setText(QCoreApplication.translate("MainWindow", u"Detect", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Enter content:", None))
        self.checkButton.setText(QCoreApplication.translate("MainWindow", u"Check", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Results", None))
        self.resultsDisplay.setText(QCoreApplication.translate("MainWindow", u"Watermark [not] detected", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"User:", None))
        self.detectBox.setPlainText("")
        self.detectAdvancedButton.setText(QCoreApplication.translate("MainWindow", u"Advanced Options", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Hashing Context", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"Z Threshold", None))
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Entropy Threshold", None))
        self.detectUploadFileButton.setText(QCoreApplication.translate("MainWindow", u"  Upload...", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u" Key file location:", None))
        self.detectHashingDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.detectEntropyDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.detectZDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u" Choose a model:", None))
        self.title2_2.setText(QCoreApplication.translate("MainWindow", u"model", None))
        self.title_2.setText(QCoreApplication.translate("MainWindow", u"Evaluate", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u" Choose a model:", None))
        self.evalUploadFileButton.setText(QCoreApplication.translate("MainWindow", u"  Upload...", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u" Choose prompts file:", None))
        self.evalButton.setText(QCoreApplication.translate("MainWindow", u"Evaluate", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Values", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Delta", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Hashing Context", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Maximum New Tokens", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Entropy", None))
        self.evalMaxDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.evalZDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Z Threshold", None))
        self.evalDeltaDisplayValue.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"Sweep values:", None))
        self.evalSweepComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"None", None))
        self.evalSweepComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Entropy", None))
        self.evalSweepComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Hashing Context", None))

        self.evalDownloadFileButton.setText(QCoreApplication.translate("MainWindow", u"  Upload...", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u" Download location:", None))
        self.evalEnt1Check.setText(QCoreApplication.translate("MainWindow", u"1.0", None))
        self.evalEnt2Check.setText(QCoreApplication.translate("MainWindow", u"2.0", None))
        self.evalEnt3Check.setText(QCoreApplication.translate("MainWindow", u"3.0", None))
        self.evalEnt4Check.setText(QCoreApplication.translate("MainWindow", u"4.0", None))
        self.evalEnt5Check.setText(QCoreApplication.translate("MainWindow", u"5.0", None))
        self.evalEnt6Check.setText(QCoreApplication.translate("MainWindow", u"6.0", None))
        self.evalHash8Check.setText(QCoreApplication.translate("MainWindow", u"8.0", None))
        self.evalHash6Check.setText(QCoreApplication.translate("MainWindow", u"6.0", None))
        self.evalHash7Check.setText(QCoreApplication.translate("MainWindow", u"7.0", None))
        self.evalHash5Check.setText(QCoreApplication.translate("MainWindow", u"5.0", None))
        self.evalHash4Check.setText(QCoreApplication.translate("MainWindow", u"4.0", None))
        self.evalHash3Check.setText(QCoreApplication.translate("MainWindow", u"3.0", None))
    # retranslateUi

