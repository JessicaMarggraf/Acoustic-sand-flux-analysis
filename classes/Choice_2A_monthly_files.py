"""
Copyright (C) 2023  INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import tkinter as tk
from tkinter import messagebox
import tkinter.filedialog
import os

class user_input_2A() :
    def __init__(self):
        self.input_map = [None]
        self.transect = None

    def open(self):
        text = ['Month [202306, 202112, ...]']
        self.value = None
        self.ws = tk.Tk()
        self.ws.title('Choose month for analysis')
        self.ws.geometry("400x200")
        self.ws.wm_attributes("-topmost", 1)
        self.ws.bind("<Return>", self.generate)

        self.tk_transect = tk.StringVar()
        i = 0
        self.label_transect = tk.Label(self.ws, text=text[i])
        self.label_transect.grid(row=i, column=0, padx=3)
        self.entry_transect = tk.Entry(self.ws, textvariable=self.tk_transect)
        self.entry_transect.grid(row=i, column=1, padx=10, pady=10)

        self.button = tk.Button(self.ws, text='Ok [Return]', command=self.generate, padx=50, pady=10)
        self.button.grid(row=i + 1, column=0, columnspan=2)
        self.ws.mainloop()

    def generate(self, event=None):
        try:
            if len(self.entry_transect.get())>0:
                # transect_list = self.entry_transect.get()
                # transect = str(transect_list)
                self.transect = str(self.entry_transect.get())           
    
            self.input_map = [self.transect]
    
            self.ws.destroy()
            print(self.input_map)
        except Exception as ex:
            messagebox.showwarning('warning', ex)
            

class choice_unit():
    def __init__(self):
        self.unit_volumes = None
        self.unit_masses = None

        self.ws = tk.Tk()
        self.ws.title('Define units')
        self.ws.geometry("175x130")
        self.ws.wm_attributes("-topmost", 1)
        self.ws.bind("<Return>", self.input_choice)

        label_volume = tk.Label(self.ws, text='Unit volumes')
        label_volume.grid(row=0, column=0)

        self.var_volume = tk.StringVar()
        rb_mL = tk.Radiobutton(self.ws, text='mL', variable=self.var_volume, value='mL')
        rb_mL.grid(row=1, column=0, sticky="W")
        rb_L = tk.Radiobutton(self.ws, text='L', variable=self.var_volume, value='L')
        rb_L.grid(row=2, column=0, sticky="W")
        rb_L.select()

        label_masses = tk.Label(self.ws, text='Unit masses')
        label_masses.grid(row=0, column=2)

        self.var_masses = tk.StringVar()
        rb_mg = tk.Radiobutton(self.ws, text='mg', variable=self.var_masses, value='mg')
        rb_mg.grid(row=1, column=2, sticky="W")
        rb_g = tk.Radiobutton(self.ws, text='g', variable=self.var_masses, value='g')
        rb_g.grid(row=2, column=2, sticky="W")
        rb_g.select()

        self.button = tk.Button(self.ws, text='Ok [Return]', command=self.input_choice, padx=20, pady=10)
        self.button.grid(row=3, column=0, columnspan=3)

        self.ws.grid_columnconfigure(1, minsize=30)
        self.ws.mainloop()

    def input_choice(self, event=None):
        self.unit_volume = self.var_volume.get()
        self.unit_masses = self.var_masses.get()

        print('unit_volume :', self.var_volume.get())
        print('unit_masses :', self.var_masses.get())
        self.ws.destroy()
        

class choice_freq():
    def __init__(self):
        self.freq = None
        # self.unit_masses = None

        self.ws = tk.Tk()
        self.ws.title('Define frequency')
        self.ws.geometry("200x200")
        self.ws.wm_attributes("-topmost", 1)
        self.ws.bind("<Return>", self.input_choice)

        self.var_volume = tk.StringVar()
        rb_mL = tk.Radiobutton(self.ws, text='400 kHz', variable=self.var_volume, value='400kHz')
        rb_mL.grid(row=1, column=0, sticky="W")
        rb_L = tk.Radiobutton(self.ws, text='1 MHz', variable=self.var_volume, value='1MHz')
        rb_L.grid(row=2, column=0, sticky="W")
        rb_L.select()

        self.button = tk.Button(self.ws, text='Ok [Return]', command=self.input_choice, padx=20, pady=10)
        self.button.grid(row=3, column=0, columnspan=3)

        self.ws.grid_columnconfigure(1, minsize=30)
        self.ws.mainloop()

    def input_choice(self, event=None):
        self.freq = self.var_volume.get()

        print('Frequency:', self.var_volume.get())
        self.ws.destroy()

def numbers_extrac(outpath) :
    copie_outpath = outpath.replace('\\', ' ').replace('_', ' ').replace('/', ' ')
    numbers_extrac = [int(s) for s in copie_outpath.split() if s.isdigit()]
    numbers_extrac = str(numbers_extrac)
    numbers_extrac = numbers_extrac.replace('[', '').replace(']', '')
    return (numbers_extrac)

def user_input_path():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath_stage = str(tk.filedialog.askdirectory(title='Define path'))
    filepath_stage = filepath_stage.replace('(', '').replace(")", '').replace("'", '').replace(",", '').replace("/", "\\")
    root.destroy()    
    return filepath_stage


def user_input_outpath():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath_stage = str(tk.filedialog.askdirectory(title='Define outpath'))
    filepath_stage = filepath_stage.replace('(', '').replace(")", '').replace("'", '').replace(",", '').replace("/", "\\")
    root.destroy()    
    return filepath_stage


def user_input_outpath_figures():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath_stage = str(tk.filedialog.askdirectory(title='Define outpath for figures'))
    filepath_stage = filepath_stage.replace('(', '').replace(")", '').replace("'", '').replace(",", '').replace("/", "\\")
    root.destroy()    
    return filepath_stage


def user_input_path_freq1():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath_stage = str(tk.filedialog.askdirectory(title='Define path intensity and background 400 kHz'))
    filepath_stage = filepath_stage.replace('(', '').replace(")", '').replace("'", '').replace(",", '').replace("/", "\\")
    root.destroy()    
    return filepath_stage


def user_input_path_freq2():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath_stage = str(tk.filedialog.askdirectory(title='Define path intensity and background 1 MHz'))
    filepath_stage = filepath_stage.replace('(', '').replace(")", '').replace("'", '').replace(",", '').replace("/", "\\")
    root.destroy()    
    return filepath_stage

def user_input_path_data():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath_stage = str(tk.filedialog.askdirectory(title='Define path of concurrent samplings, missing and deleted data and station data'))
    filepath_stage = filepath_stage.replace('(', '').replace(")", '').replace("'", '').replace(",", '').replace("/", "\\")
    root.destroy()    
    return filepath_stage

class user_input_events_dates() :
    def __init__(self):
        self.input_map = [None, None]

        self.transect = None
        self.nav_ref = None

    def open(self):
        text = ['Start date [DD/MM/YYYY HH:MM]', 'End date [DD/MM/YYYY HH:MM]']
        self.value = None
        self.ws = tk.Tk()
        self.ws.title('Choose start and end date of event')
        self.ws.geometry("400x250")
        self.ws.wm_attributes("-topmost", 1)
        self.ws.bind("<Return>", self.generate)

        self.tk_transect = tk.StringVar()
        self.tk_nav_ref = tk.StringVar()
        i = 0
        self.label_transect = tk.Label(self.ws, text=text[i])
        self.label_transect.grid(row=i, column=0, padx=3)
        self.entry_transect = tk.Entry(self.ws, textvariable=self.tk_transect)
        self.entry_transect.grid(row=i, column=1, padx=10, pady=10)
        i += 1
        self.label_nav_ref = tk.Label(self.ws, text=text[i])
        self.label_nav_ref.grid(row=i, column=0, padx=3)
        self.entry_nav_ref = tk.Entry(self.ws, textvariable=self.tk_nav_ref)
        self.entry_nav_ref.grid(row=i, column=1, padx=10, pady=10)

        self.button = tk.Button(self.ws, text='Ok [Return]', command=self.generate, padx=50, pady=10)
        self.button.grid(row=i + 1, column=0, columnspan=2)
        self.ws.mainloop()

    def generate(self, event=None):
        try:
            if len(self.entry_transect.get())>0:
                self.transect = str(self.entry_transect.get())
                # transect_list = self.entry_transect.get()
                # transect = [int(s) for s in transect_list.split(',')]
                # self.transect = list(map(int, transect))
            if len(self.entry_nav_ref.get()) > 0:
                self.nav_ref = str(self.entry_nav_ref.get())
    
            self.input_map = [self.transect, self.nav_ref]
    
            self.ws.destroy()
            print(self.input_map)
        except Exception as ex:
            messagebox.showwarning('warning', ex)
            

class ask_float():
    def __init__(self, text: str = '', title: str = '', default_value:str=''):
        self.value = None
        self.ws = tk.Tk()
        self.ws.title(title)
        self.ws.geometry("250x100")
        self.ws.wm_attributes("-topmost", 1)
        self.ws.bind("<Return>", self.generate)
        self.num1 = tk.StringVar(value=default_value)
        self.label = tk.Label(self.ws, text=text)
        self.label.pack()
        self.entry1 = tk.Entry(self.ws, textvariable=self.num1)
        self.entry1.pack()
        self.button = tk.Button(self.ws, text='Ok [Return]', command=self.generate, padx=50, pady=10)
        self.button.pack()
        self.ws.mainloop()

    def generate(self, event=None):
        try:
            result = float(self.num1.get())
            self.value = result
            self.ws.destroy()
            print(self.value)
        except Exception as ex:
            messagebox.showwarning('warning', ex)
        
