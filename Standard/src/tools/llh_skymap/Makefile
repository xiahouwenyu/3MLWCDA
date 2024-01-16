# Author: Zhiguo Yao <zhiguo.yao@ihep.ac.cn>, 2011/06/01
include Makefile.arch

#------------------------------------------------------------------------------

SHELL          = /bin/bash
DEBUG          = no


MAINSRCS       = Add_UserInfo.cc
MAINOBJS       = $(MAINSRCS:.cc=.o)
MAINEXES       = $(MAINOBJS:.o=)

CLASSSRCS      = 
CLASSOBJS      = $(CLASSSRCS:.cc=.o)
DICTOBJS       = $(CLASSOBJS:.o=Dict.o)
DICTSRCS       = $(DICTOBJS:.o=.cc)
LINKDEFS       = $(CLASSOBJS:.o=LinkDef.h)

MISCSRCS       = #inc_astro/hpatimer.cc inc_astro/papi.cc  wcdaevent.cc dataCut/EventCut.cc
MISCOBJS       = $(MISCSRCS:.cc=.o)

PROGRAMS       = $(MAINEXES)
OBJS           = $(CLASSOBJS) $(DICTOBJS) $(MISCOBJS)
ALLOBJS        = $(MAINOBJS) $(OBJS)

LIBS          += -lMinuit -lz -L$(SLALIB_LIBDIR) -lsla 
LIBS          += $(shell pkg-config  --libs cfitsio)
LIBS          += $(shell pkg-config  --libs healpix_cxx)

ifeq ($(DEBUG),yes)
CXXFLAGS      := -g -DDEBUG $(filter-out -O -O1 -O2 -O3 -O4,$(CXXFLAGS))
LDFLAGS       := $(filter-out -O -O1 -O2 -O3 -O4,$(LDFLAGS))
endif

CXXFLAGS      += -I$(SLALIB_INCDIR)
CXXFLAGS      += -D_MAIN_
CXXFLAGS      += $(shell pkg-config --cflags cfitsio)
CXXFLAGS      += $(shell pkg-config --cflags healpix_cxx)
CXXFLAGS      += -I./inc
#------------------------------------------------------------------------------
.SUFFIXES: .cc .o .so .h .d
.PHONY: clean distclean test

all: $(PROGRAMS)

ifeq ($(findstring $(MAKECMDGOALS),clean distclean test),)
include $(ALLOBJS:.o=.d)
endif

test: $(PROGRAMS) exam.dat
	./$< exam.dat exam.root

clean:
#@rm -f $(OBJS) core *.so lib*.a *.d *.d.[0-9]* *.o *Dict.* *LinkDef.h *.pcm
	@rm -f $(OBJS) core *.so lib*.a *.d *.d.[0-9]* *.o 

distclean: clean
	@rm -f $(PROGRAMS) *.root *.eps *.dat

$(PROGRAMS): % : %.o $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LIBS) -o $@
	@echo "$@ done"

$(ALLOBJS): %.o : %.cc %.d
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(DICTSRCS): %Dict.cc: %.h %LinkDef.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

$(LINKDEFS): %LinkDef.h: %.d
	@echo "#ifdef __CINT__" > $@; \
	echo "" >> $@; \
	echo "#pragma link off all globals;" >> $@; \
	echo "#pragma link off all classes;" >> $@; \
	echo "#pragma link off all functions;" >> $@; \
	echo "" >> $@; \
	echo "#pragma link C++ class $*+;" >> $@; \
	echo "" >> $@; \
	echo "#endif" >> $@

$(ALLOBJS:.o=.d): %.d: %.cc
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	DIR=$$(dirname $*)"/"; BASE=$$(basename $*); \
	[ "$$DIR" != "./" ] || DIR=""; \
	sed "s,\($$BASE\)\.o[ :]*,$$DIR\1.o $@: ,g" < $@.$$$$ > $@; \
	rm -f $@.$$$$

