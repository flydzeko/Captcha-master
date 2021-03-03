# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Jun 17 2015)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
from wx.lib.embeddedimage import PyEmbeddedImage
import train_captcha
import recog_test
from threading import Thread
import index
import inspect
import ctypes
import sample

ident1=0
ident2=0


class MyFrame1 ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u" 验证码识别工具", pos = wx.DefaultPosition, size = wx.Size( 650,360 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_INACTIVECAPTION ))
		
		bSizer1 = wx.BoxSizer( wx.VERTICAL )

		self.myImage = wx.StaticBitmap(self, -1,size=(630,50))
		bmp = wx.Bitmap("./res/head4.jpg", wx.BITMAP_TYPE_JPEG)
		self.myImage.SetBitmap(bmp)
		
		bSizer1.Add( self.myImage, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		bSizer2 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_textCtrl2 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 250,-1 ), 0 )
		bSizer2.Add( self.m_textCtrl2, 0, wx.ALL, 5 )
		
		self.m_button1 = wx.Button( self, wx.ID_ANY, u"训练样本",  wx.DefaultPosition, wx.Size(60, -1), 0)
		bSizer2.Add( self.m_button1, 0, wx.ALL, 5 )
		self.Bind(wx.EVT_BUTTON, self.OnButtonClick1,self.m_button1)

		self.m_button2 = wx.Button( self, wx.ID_ANY, u"开始训练",  wx.DefaultPosition, wx.Size(60, -1), 0)
		bSizer2.Add( self.m_button2, 0, wx.ALL, 5 )
		self.Bind(wx.EVT_BUTTON, self.OnButtonClick2, self.m_button2)

		self.m_button6 = wx.Button(self, wx.ID_ANY, u"停止训练", wx.DefaultPosition, wx.Size(60, -1), 0)
		bSizer2.Add(self.m_button6, 0, wx.ALL, 5)
		self.Bind(wx.EVT_BUTTON, self.OnButtonClick6, self.m_button6)

		bSizer1.Add(bSizer2, 1, wx.ALIGN_CENTER_HORIZONTAL, 5)
		
		bSizer4 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_textCtrl3 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 255,-1 ), 0 )
		bSizer4.Add( self.m_textCtrl3, 0, wx.ALL, 5 )
		
		self.m_button4 = wx.Button( self, wx.ID_ANY, u"测试样本", wx.DefaultPosition, wx.Size( 95,-1 ), 0 )
		bSizer4.Add( self.m_button4, 0, wx.ALL, 5 )
		self.Bind(wx.EVT_BUTTON, self.OnButtonClick4, self.m_button4)
		
		self.m_button5 = wx.Button( self, wx.ID_ANY, u"开始测试", wx.DefaultPosition, wx.Size( 95,-1 ), 0 )
		bSizer4.Add( self.m_button5, 0, wx.ALL, 5 )
		self.Bind(wx.EVT_BUTTON, self.OnButtonClick5, self.m_button5)

		bSizer1.Add(bSizer4, 1, wx.ALIGN_CENTER, 5)
		
		self.m_textCtrl1 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 480,140 ), wx.TE_MULTILINE )
		bSizer1.Add( self.m_textCtrl1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.SetSizer( bSizer1 )
		self.Layout()
		
		self.Centre( wx.BOTH )

	class TrainThread(Thread):
		def __init__(self):
			Thread.__init__(self)
			self.start()  # start the thread
			global ident1
			ident1 = self.ident

		def run(self):
			train_captcha.main()

	class UpdateThread(Thread):
		def __init__(self,obj):
			Thread.__init__(self)
			self.start()
			self.obj=obj
			global ident2
			ident2=self.ident

		def run(self):
			f1 = open('temp1.txt', 'r')
			while index.fileindex1 != -1:
				if index.fileindex1 == 1:
					self.obj.m_textCtrl1.AppendText(f1.readline())
					index.fileindex1 = 0
			f1.close()

	class TestThread(Thread):
		def __init__(self):
			Thread.__init__(self)
			self.start()  # start the thread

		def run(self):
			recog_test.main()

	class DisplayThread(Thread):
		def __init__(self,obj):
			Thread.__init__(self)
			self.start()
			self.obj=obj

		def run(self):
			f2 = open('temp2.txt', 'r')
			f3 = open('temp3.txt', 'r')
			while index.fileindex2 != -1:
				if index.fileindex2 == 1:
					self.obj.m_textCtrl1.AppendText(f2.readline())
					# self.obj.m_textCtrl1.AppendText("$\n")
					index.fileindex2 = 0
			self.obj.m_textCtrl1.AppendText(f2.readline())
			self.obj.m_textCtrl1.AppendText(f3.read())
			f2.close()
			f3.close()

	def OnButtonClick1(self, event):
		dlg1 = wx.DirDialog(self, u"请选择训练样本保存路径", style=wx.DD_DEFAULT_STYLE)
		if dlg1.ShowModal() == wx.ID_OK:
			path=dlg1.GetPath()
			sample.train_image_dir=path
			self.m_textCtrl2.AppendText(path)


	def OnButtonClick2(self, event):
		self.TrainThread()
		btn = event.GetEventObject()
		btn.Disable()
		self.m_textCtrl1.AppendText("Start training.....\n")
		self.UpdateThread(self)
		# t = threading.Thread(target=self.UpdateText())
		# t.setDaemon(True)  # 设置为后台线程，这里默认是False，设置为True之后则主线程不用等待子线程
		# t.start()


	def OnButtonClick4(self, event):
		dlg4 = wx.DirDialog(self, u"请选择测试样本保存路径", style=wx.DD_DEFAULT_STYLE)
		if dlg4.ShowModal() == wx.ID_OK:
			path=dlg4.GetPath()
			sample.test_image_dir=path
			self.m_textCtrl3.AppendText(path)

	def OnButtonClick5(self, event):
		self.TestThread()
		self.m_textCtrl1.AppendText("Start test......\n")
		self.DisplayThread(self)

	def OnButtonClick6(self, event):
		global  ident1
		global  ident2
		self._async_raise(ident1, SystemExit)
		self._async_raise(ident2, SystemExit)
		self.m_button2.Enable()
		self.m_textCtrl1.AppendText("Training has stoped!\n")
		f1 = open('temp1.txt', 'w')
		f1.close()


	# 终止线程
	def _async_raise(self,tid, exctype):
		"""raises the exception, performs cleanup if needed"""
		tid = ctypes.c_long(tid)
		if not inspect.isclass(exctype):
			exctype = type(exctype)
		res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
		if res == 0:
			raise ValueError("invalid thread id")
		elif res != 1:
			# """if it returns a number greater than one, you're in trouble,
			# and you should call it again with exc=NULL to revert the effect"""
			ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
			raise SystemError("PyThreadState_SetAsyncExc failed")

	'''
	def UpdateText(self):
		f1 = open('temp1.txt', 'r')
		while index.fileindex1 != -1:
			if index.fileindex1 == 1:
				self.m_textCtrl1.AppendText(f1.readline())
				index.fileindex1 = 0
		f1.close()
	'''

	def __del__( self ):
		pass
	
if __name__ == '__main__':
    app = wx.App()
    main_win = MyFrame1(None)
    main_win.Show()
    app.MainLoop()
