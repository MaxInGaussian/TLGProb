"""
Created on Sun Oct  28 21:37:50 2015
@author: Max W. Y. Lam
"""
import os
import sys
import time
import shutil
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException


class bot(object):

    br_dict, idle_brs = {}, []

    # Bot Command Types
    GOTO, FINDC, FINDI, FINDL, FINDN, FINDT, FINDX = 0, 1, 2, 3, 4, 5, 6

    def __init__(self, browser="phantomjs", buf=1., trials=10):
        self.dr_path = shutil.which(browser)
        self.buf = buf
        self.trials = trials
        self.br_dict[0] = webdriver.PhantomJS(executable_path=self.dr_path)
        self.idle_brs.append(0)

    def _remove_br(self, br_id):
        if(br_id not in self.idle_brs):
            print("Brower", br_id, "is busy! Brower Removal Error")
            return
        self.br_dict[br_id].close()
        self.br_dict[br_id] = None
    
    def _idle_br(self, br_id):
        self.idle_brs.append(br_id)

    def _get_any_idle_br_id(self):
        if(len(self.idle_brs) == 0):
            print("All browers are busy! Add New Session")
            br_id = 0
            if(len(self.br_dict) > 0):
                br_id = max(self.br_dict.keys()) + 1
            self.idle_brs.append(br_id)
            self.br_dict[br_id] = webdriver.PhantomJS(executable_path=self.dr_path)
        return self.idle_brs[0]

    def _new_command(self, br_id, cmd_argv):
        assert isinstance(cmd_argv, list) or isinstance(cmd_argv, tuple),\
            "Command Argument Error"
        assert len(cmd_argv) > 0, "No Command Argument!"
        if(br_id in self.idle_brs):
            self.idle_brs.remove(br_id)
        cmd_type = cmd_argv[0]
        cmd_status, cmd_return = False, None
        if(cmd_type <= self.GOTO):
            url = cmd_argv[1]
            print("loading to", url)
            self.br_dict[br_id].get(url)
            cmd_status = True
            if(len(cmd_argv) > 2):
                cnts_type = cmd_argv[2]
                cnts_str = cmd_argv[3]
                cmd_return = self._find_elements(
                    self.br_dict[br_id], cnts_type, cnts_str)
                cmd_status = (len(cmd_return) > 0)
        elif(cmd_type <= self.FINDX):
            find_str = cmd_argv[1]
            cmd_return = self._find_elements(
                self.br_dict[br_id], cmd_type, find_str)
            cmd_status = (len(cmd_return) > 0)
            if(len(cmd_argv) > 2 and len(cmd_return) > 0):
                for i in range(2, len(cmd_argv), 2):
                    find_type = cmd_argv[i]
                    find_str = cmd_argv[i+1]
                    cmd_return = self._find_elements(
                        cmd_return[0], find_type, find_str)
        sys.stdout.flush()
        return (cmd_status, cmd_return)

    def _find_elements(self, target, find_type, find_str):
        if(find_type == self.FINDC):
            return self._find_class(target, find_str)
        elif(find_type == self.FINDI):
            return self._find_id(target, find_str)
        elif(find_type == self.FINDL):
            return self._find_link_text(target, find_str)
        elif(find_type == self.FINDN):
            return self._find_name(target, find_str)
        elif(find_type == self.FINDT):
            return self._find_tag(target, find_str)
        elif(find_type == self.FINDX):
            return self._find_xpath(target, find_str)
        sys.stdout.flush()

    def _find_class(self, target, c):
        for _ in range(self.trials):
            sys.stdout.flush()
            try:
                e = target.find_elements_by_class_name(c)
                t = e[0]
                return e
            except Exception:
                time.sleep(self.buf)
                continue
        return []

    def _find_id(self, target, i):
        for _ in range(self.trials):
            sys.stdout.flush()
            try:
                e = target.find_elements_by_id(i)
                t = e[0]
                return e
            except Exception:
                time.sleep(self.buf)
                continue
        return []

    def _find_link_text(self, target, l):
        for _ in range(self.trials):
            sys.stdout.flush()
            try:
                e = target.find_elements_by_partial_link_text(l)
                t = e[0]
                return e
            except Exception:
                time.sleep(self.buf)
                continue
        return []

    def _find_name(self, target, n):
        for _ in range(self.trials):
            sys.stdout.flush()
            try:
                e = target.find_elements_by_name(n)
                t = e[0]
                return e
            except Exception:
                time.sleep(self.buf)
                continue
        return []

    def _find_tag(self, target, t):
        for _ in range(self.trials):
            sys.stdout.flush()
            try:
                e = target.find_elements_by_tag_name(t)
                t = e[0]
                return e
            except Exception:
                time.sleep(self.buf)
                continue
        return []

    def _find_xpath(self, target, x):
        for _ in range(self.trials):
            sys.stdout.flush()
            try:
                e = target.find_elements_by_xpath(x)
                t = e[0]
                return e
            except Exception:
                time.sleep(self.buf)
                continue
        return []

    def _close_all_br(self):
        for br in self.br_dict.values():
            br.close()
            br.quit()
